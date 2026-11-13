"""Export a model set to ONNX and compile it for a Rockchip NPU.

    python -m neuronte_npu.convert --manifest M --case NAME --cell 16x32 \
        --batch 256 --out DIR [--dense] [--platform rk3588] [--onnx-only]

Writes NAME-WxH-bB-{g|d}-L1.onnx, .rknn (unless --onnx-only) and .meta, plus
a .l1 (host layer 1: character weights, biases and position term) or .pos
(position input) sidecar for the forms that need one. The .meta text file
tells the C backend how to build inputs and assemble outputs. Compilation
uses RKNN-Toolkit2 and is float16 (no quantization).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import graph, layout as L, nrt

META_VERSION = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_meta(path, lay, batch, grouped, members, pos_file=None, l1_file=None):
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


def compile_rknn(onnx_path, rknn_path, platform, level, log):
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    try:
        if rknn.config(target_platform=platform, optimization_level=level) != 0:
            raise RuntimeError("rknn.config failed")
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            raise RuntimeError("rknn.load_onnx failed")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError("rknn.build failed")
        if rknn.export_rknn(str(rknn_path)) != 0:
            raise RuntimeError("rknn.export_rknn failed")
    finally:
        rknn.release()
    print(f"compiled {rknn_path}", file=log)


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
    args = ap.parse_args(argv)
    w, h = map(int, args.cell.lower().split("x"))
    if not 1 <= args.batch <= 4096:
        raise SystemExit("batch must be 1..4096")
    lay = L.from_manifest(args.manifest, args.case, w, h, args.samples)
    grouped = not args.dense
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.case}-{w}x{h}-b{args.batch}-{'g' if grouped else 'd'}-{args.l1}"
    model = graph.build(lay, args.batch, grouped, args.l1)
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
               l1_file)
    if not args.onnx_only:
        compile_rknn(onnx_path, out / f"{stem}.rknn", args.platform, args.optimization_level, sys.stderr)
    record = {"stem": stem, "layout": lay.describe(), "batch": args.batch, "conv": "grouped" if grouped else "dense",
              "platform": args.platform, "l1": args.l1, "optimization_level": args.optimization_level,
              "toolkit": toolkit_version(), "onnx_sha256": digest(onnx_path),
              "meta_sha256": digest(out / f"{stem}.meta")}
    if pos_file:
        record["pos_sha256"] = digest(out / pos_file)
    if l1_file:
        record["l1_sha256"] = digest(out / l1_file)
    if not args.onnx_only:
        record["rknn_sha256"] = digest(out / f"{stem}.rknn")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
