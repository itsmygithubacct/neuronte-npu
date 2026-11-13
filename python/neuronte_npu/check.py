"""Check exported graphs against the reference CPU library, before compiling.

    python -m neuronte_npu.check --manifest M --lib libneuronte.so \
        [--case NAME ...] [--cell WxH ...] [--l1 host] [--dense]

Builds each graph, runs it with ONNX Runtime in float32 over the 256 Braille
scalars, and compares the coverage with neuronte_coverage for the same tiles.
Prints one JSON record per case and cell: the largest absolute difference and
the number of pixels whose 0.5 threshold differs. The NPU computes in fp16, so
this checks the graph and the layout, not NPU precision.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import cpu, graph, layout as L

CELLS = ("8x16", "16x32", "48x96")
SCALARS = [0x2800 + i for i in range(256)]


def onnx_coverage(lay, scalars, l1, grouped):
    import onnxruntime as ort
    model = graph.build(lay, len(scalars), grouped, l1)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    if l1 == "host":
        feeds = {"h1": graph.host_input(lay, scalars)}
    else:
        feeds = {"code": graph.code_input(lay, scalars)}
        if l1 in graph.POS_INPUTS:
            batch = 1 if l1 == "input-shared" else len(scalars)
            feeds["pos"] = graph.position_input(lay, batch)
    return graph.assemble(lay, session.run(None, feeds)[0])


def check(manifest, lib, case, cell, l1="host", grouped=True):
    w, h = map(int, cell.split("x"))
    lay = L.from_manifest(manifest, case, w, h)
    data = json.loads(Path(manifest).read_text())
    members = next(c for c in data["cases"] if c["name"] == case)["members"]
    paths = [Path(manifest).parent / data["models"][m]["file"] for m in members]
    ref = np.asarray(lib.coverage(lay, paths, SCALARS))
    got = onnx_coverage(lay, SCALARS, l1, grouped)
    return {"case": case, "cell": cell, "l1": l1, "conv": "grouped" if grouped else "dense",
            "max_abs": float(np.abs(got - ref).max()),
            "binary_flips": int(((got >= 0.5) != (ref >= 0.5)).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lib", required=True, help="path to libneuronte.so")
    ap.add_argument("--case", action="append", help="default: every case")
    ap.add_argument("--cell", action="append", help=f"default: {' '.join(CELLS)}")
    ap.add_argument("--l1", default="host", choices=graph.L1_ACTIVATIONS)
    ap.add_argument("--dense", action="store_true")
    args = ap.parse_args(argv)
    lib = cpu.Library(args.lib)
    cases = args.case or [c["name"] for c in json.loads(Path(args.manifest).read_text())["cases"]]
    worst = 0
    for case in cases:
        for cell in args.cell or CELLS:
            rec = check(args.manifest, lib, case, cell, args.l1, not args.dense)
            worst = max(worst, rec["binary_flips"])
            print(json.dumps(rec), flush=True)
    raise SystemExit(1 if worst else 0)


if __name__ == "__main__":
    main()
