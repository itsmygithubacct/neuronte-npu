"""Export a model set to ONNX and compile it for a Rockchip NPU.

    python -m neuronte_npu.convert --manifest M --case NAME --cell 16x32 \
        --batch 256 --out DIR [--dense] [--platform rk3588] [--onnx-only] \
        [--pool avg|conv] [--dtype fp16|w8a8 [--quant-algorithm normal|mmse]] \
        [--refine MARGIN]

Writes NAME-WxH-bB-{g|d}-L1[-pconv][-w8a8-ALGO].onnx, .rknn (unless
--onnx-only) and .meta, plus a .l1 (host layer 1: character weights, biases
and position term) or .pos (position input) sidecar for the forms that need
one. The .meta text file tells the C backend how to build inputs and
assemble outputs. Compilation uses RKNN-Toolkit2, float16 by default.

--dtype w8a8 quantizes weights and activations to int8 (host layer-1 form
only). Calibration data is the graph input for every scalar of the Braille
block U+2800..U+28FF, the whole input domain of these models, in batches of
the model's batch size; its sha256 is recorded. Quantization error depends
on the model: score every quantized model against the CPU before use, and
give the backend a refine_margin above its largest error.

--refine MARGIN writes the margin into the .meta ("refine MARGIN"): the
backend then recomputes, on the CPU, every tile with a pixel within MARGIN of
0.5 unless the caller sets refine_margin itself. Use it for a model whose
output matches the CPU's only with refinement.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from . import graph, layout as L, nrt

META_VERSION = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_meta(path, lay, batch, grouped, members, pos_file=None, l1_file=None, refine=0.0):
    lines = [f"neuronte-npu-meta {META_VERSION}",
             f"name {lay.name}",
             f"cell {lay.cell_w} {lay.cell_h}",
             f"samples {lay.samples}",
             f"batch {batch}",
             f"hidden {lay.hidden}",
             f"conv {'grouped' if grouped else 'dense'}",
             f"tiles {len(lay.tiles)}"]
    for i, (t, member) in enumerate(zip(lay.tiles, members, strict=True)):
        mask = -1 if t.mask is None else t.mask
        lines.append(f"tile {i} {mask} {t.x} {t.y} {t.w} {t.h} {member['file']} {member['sha256']} "
                     + " ".join(str(m) for m in t.net.masks))
    lines.append(f"blocks {len(lay.blocks)} {lay.block_w} {lay.block_h}")
    for g, b in enumerate(lay.blocks):
        lines.append(f"block {g} {b.tile} {b.x} {b.y}")
    lines.append(f"input 1 {len(lay.tiles) * nrt.CODE} {batch} 1")
    lines.append(f"output 1 {len(lay.blocks)} {batch} {lay.pixels_per_block}")
    if pos_file:
        lines.append(f"pos {pos_file} {len(lay.blocks) * lay.hidden} {lay.samples_per_block}")
    if l1_file:
        lines.append(f"host-l1 {l1_file} {len(lay.blocks) * lay.hidden} {len(lay.tiles) * nrt.CODE} "
                     f"{lay.samples_per_block}")
    if refine:
        lines.append(f"refine {refine:g}")
    Path(path).write_text("\n".join(lines) + "\n")


def toolkit_version():
    try:
        from importlib.metadata import version
        return version("rknn-toolkit2")
    except Exception:
        return None


def members_of(manifest, case_name):
    data = json.loads(Path(manifest).read_text())
    case = next(c for c in data["cases"] if c["name"] == case_name)
    return [data["models"][m] for m in case["members"]]


def write_calibration(lay, batch, out_dir):
    """Graph inputs for every Braille scalar, as .npy files and a dataset list.
    Returns (dataset path, sha256 over the arrays in order)."""
    import numpy as np
    first, last = L.BRAILLE_SPAN
    scalars = list(range(first, last + 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    lines = []
    for k in range(0, len(scalars), batch):
        chunk = scalars[k:k + batch]
        chunk += [chunk[0]] * (batch - len(chunk))  # pad like the backend does
        x = graph.host_input(lay, chunk)
        h.update(np.ascontiguousarray(x).tobytes())
        path = out_dir / f"{k // batch:04d}.npy"
        np.save(path, x)
        lines.append(str(path.resolve()))
    dataset = out_dir / "dataset.txt"
    dataset.write_text("\n".join(lines) + "\n")
    return dataset, h.hexdigest()


def sim_check(rknn, onnx_path, dataset, lay):
    """Run the compiled model in the toolkit's simulator on every dataset
    input and compare its coverage with the float ONNX graph (which matches
    the CPU library to about 1e-5). The simulator reproduces the NPU's
    quantized arithmetic closely, so this screens a quantized model before
    it reaches a board."""
    import numpy as np
    import onnxruntime as ort
    if rknn.init_runtime() != 0:
        raise RuntimeError("rknn.init_runtime (simulator) failed")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst, flips, near = 0.0, 0, 0
    for path in Path(dataset).read_text().split():
        x = np.load(path)
        ref = graph.assemble(lay, sess.run(None, {"h1": x})[0])
        got = graph.assemble(lay, rknn.inference(inputs=[x], data_format=["nchw"])[0])
        worst = max(worst, float(np.abs(got - ref).max()))
        flips += int(((got >= 0.5) != (ref >= 0.5)).sum())
        near += int((np.abs(ref - 0.5) < 0.05).sum())
    return {"sim_max_abs": worst, "sim_flips": flips, "near_threshold_0.05": near}


def compile_rknn(onnx_path, rknn_path, platform, level, log, dtype="fp16", algorithm="normal", dataset=None,
                 check_lay=None):
    """Compile; with check_lay (a layout) and a dataset, also return sim_check()."""
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    result = None
    try:
        quant = dtype != "fp16"
        extra = {"quantized_dtype": dtype, "quantized_algorithm": algorithm} if quant else {}
        if rknn.config(target_platform=platform, optimization_level=level, **extra) != 0:
            raise RuntimeError("rknn.config failed")
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            raise RuntimeError("rknn.load_onnx failed")
        if rknn.build(do_quantization=quant, dataset=str(dataset) if quant else None) != 0:
            raise RuntimeError("rknn.build failed")
        if rknn.export_rknn(str(rknn_path)) != 0:
            raise RuntimeError("rknn.export_rknn failed")
        if check_lay is not None and dataset is not None:
            result = sim_check(rknn, onnx_path, dataset, check_lay)
    finally:
        rknn.release()
    print(f"compiled {rknn_path}", file=log)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--cell", required=True, help="WxH, e.g. 16x32")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--samples", type=int, default=0, help="per axis; 0 = automatic from cell width")
    ap.add_argument("--dense", action="store_true", help="block-diagonal dense convs instead of grouped")
    ap.add_argument("--platform", default="rk3588")
    ap.add_argument("--l1", default="host", choices=graph.L1_ACTIVATIONS, help="layer-1 form (see graph.build)")
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx-only", action="store_true")
    ap.add_argument("--pool", default="avg", choices=("avg", "conv"),
                    help="sample averaging: AveragePool, or the same as a strided depthwise conv")
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "w8a8"))
    ap.add_argument("--quant-algorithm", default="normal", choices=("normal", "mmse", "kl_divergence"))
    ap.add_argument("--refine", type=float, default=0.0, metavar="MARGIN",
                    help="refinement margin the backend applies to this model by default (0 = none)")
    ap.add_argument("--sim-check", action="store_true",
                    help="score the compiled model in the toolkit simulator over every Braille scalar")
    args = ap.parse_args(argv)
    if not 0.0 <= args.refine < 0.5:
        raise SystemExit("--refine must be in [0, 0.5)")
    if args.dtype != "fp16" and args.l1 != "host":
        raise SystemExit("--dtype w8a8 needs --l1 host (the backend quantizes h1 itself)")
    w, h = map(int, args.cell.lower().split("x"))
    if not 1 <= args.batch <= 4096:
        raise SystemExit("batch must be 1..4096")
    lay = L.from_manifest(args.manifest, args.case, w, h, args.samples)
    grouped = not args.dense
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.case}-{w}x{h}-b{args.batch}-{'g' if grouped else 'd'}-{args.l1}"
    if args.pool != "avg":
        stem += f"-p{args.pool}"
    if args.dtype != "fp16":
        stem += f"-{args.dtype}-{args.quant_algorithm}"
    model = graph.build(lay, args.batch, grouped, args.l1, args.pool)
    onnx_path = out / f"{stem}.onnx"
    onnx_path.write_bytes(model.SerializeToString())
    pos_file = None
    if args.l1 in graph.POS_INPUTS:
        pos_file = f"{stem}.pos"
        graph.position_constant(lay).astype("<f4").tofile(out / pos_file)
    l1_file = None
    if args.l1 == "host":
        l1_file = f"{stem}.l1"
        w, b, p = graph.host_layer1(lay)
        with open(out / l1_file, "wb") as f:
            for a in (w, b, p):
                f.write(a.astype("<f4").tobytes())
    write_meta(out / f"{stem}.meta", lay, args.batch, grouped, members_of(args.manifest, args.case), pos_file,
               l1_file, args.refine)
    calib_sha = check = None
    if args.sim_check and args.l1 != "host":
        raise SystemExit("--sim-check supports the host layer-1 form")
    if not args.onnx_only:
        with tempfile.TemporaryDirectory(prefix=f".{stem}.calib-", dir=out) as tmp:
            dataset = None
            if args.dtype != "fp16" or args.sim_check:  # inputs are large; only their sha256 is kept
                dataset, calib_sha = write_calibration(lay, args.batch, Path(tmp))
            check = compile_rknn(onnx_path, out / f"{stem}.rknn", args.platform, args.optimization_level,
                                 sys.stderr, args.dtype, args.quant_algorithm, dataset,
                                 lay if args.sim_check else None)
    record = {"stem": stem, "layout": lay.describe(), "batch": args.batch, "conv": "grouped" if grouped else "dense",
              "platform": args.platform, "l1": args.l1, "pool": args.pool, "dtype": args.dtype, "refine": args.refine,
              "optimization_level": args.optimization_level,
              "toolkit": toolkit_version(), "onnx_sha256": digest(onnx_path),
              "meta_sha256": digest(out / f"{stem}.meta")}
    if args.dtype != "fp16":
        record.update(quant_algorithm=args.quant_algorithm, calibration="U+2800..U+28FF host_input",
                      calibration_sha256=calib_sha)
    if check:
        record.update(check)
    if pos_file:
        record["pos_sha256"] = digest(out / pos_file)
    if l1_file:
        record["l1_sha256"] = digest(out / l1_file)
    if not args.onnx_only:
        record["rknn_sha256"] = digest(out / f"{stem}.rknn")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
