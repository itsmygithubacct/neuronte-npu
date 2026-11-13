#!/usr/bin/env python3
"""Compare coverage dumps from neuronte_npu_bench --coverage-out.

    compare.py CELL GOT.f32 [--cpu CPU.f32] [--truth TRUTH.npy] [--json]

A dump is float32 [256, h, w]: the coverage of the Braille scalars
U+2800..U+28FF in order. Against a CPU dump (normally the reference
library's own dump from --mode cpu) it reports the largest and mean absolute
difference, the binary flips (pixels on opposite sides of 0.5), the glyphs
with a flip and whether the dumps are bit-identical. Against teacher masks
([256, h, w], any dtype) it reports exact glyphs, wrong pixels and IoU with
the binary rule coverage > 0.5, the same definitions as kilix-bench's
Braille quality(). Exits 1 when GOT has binary flips against CPU.
"""
import argparse
import json
import sys

import numpy as np

GLYPHS = 256


def load(path, w, h):
    a = np.fromfile(path, dtype=np.float32)
    if a.size != GLYPHS * w * h:
        raise SystemExit(f"{path}: {a.size} floats, expected {GLYPHS * w * h} ({GLYPHS} x {h} x {w})")
    if not np.isfinite(a).all():
        raise SystemExit(f"{path}: non-finite values")
    return a.reshape(GLYPHS, h, w)


def vs_cpu(got, cpu):
    d = np.abs(got - cpu)
    flips = (got > 0.5) != (cpu > 0.5)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "binary_flips": int(flips.sum()),
            "glyphs_with_flips": int(flips.any((1, 2)).sum()), "bit_identical": bool(np.array_equal(got, cpu))}


def vs_truth(cov, truth):
    actual, wanted = cov > 0.5, truth > 0.5
    inter, union = (actual & wanted).sum((1, 2)), (actual | wanted).sum((1, 2))
    iou = np.divide(inter, union, out=np.ones(len(cov)), where=union != 0)
    wrong = (actual != wanted).sum((1, 2))
    return {"exact_glyphs": int((wrong == 0).sum()), "glyphs": len(cov), "wrong_pixels": int(wrong.sum()),
            "iou_mean": float(iou.mean()), "iou_min": float(iou.min()),
            "coverage_mae": float(np.abs(cov - truth).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cell", help="WxH, e.g. 16x32")
    ap.add_argument("got")
    ap.add_argument("--cpu")
    ap.add_argument("--truth")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    w, h = map(int, a.cell.lower().split("x"))
    got = load(a.got, w, h)
    rec = {"file": a.got, "cell": a.cell}
    if a.cpu:
        rec["vs_cpu"] = vs_cpu(got, load(a.cpu, w, h))
    if a.truth:
        truth = np.load(a.truth, allow_pickle=False).astype(np.float32)
        if truth.shape != got.shape:
            raise SystemExit(f"{a.truth}: shape {truth.shape}, expected {got.shape}")
        rec["truth"] = vs_truth(got, truth)
        if a.cpu:
            rec["cpu_truth"] = vs_truth(load(a.cpu, w, h), truth)
    if a.json:
        print(json.dumps(rec))
    else:
        if "vs_cpu" in rec:
            c = rec["vs_cpu"]
            print(f"vs CPU: max {c['max_abs']:.3g} mean {c['mean_abs']:.3g} flips {c['binary_flips']} "
                  f"in {c['glyphs_with_flips']} glyphs{' (bit-identical)' if c['bit_identical'] else ''}")
        if "truth" in rec:
            t = rec["truth"]
            line = f"vs truth: exact {t['exact_glyphs']}/{t['glyphs']} wrong px {t['wrong_pixels']} IoU min {t['iou_min']:.3f}"
            if "cpu_truth" in rec:
                line += f" (CPU: exact {rec['cpu_truth']['exact_glyphs']}, wrong px {rec['cpu_truth']['wrong_pixels']})"
            print(line)
    return 1 if rec.get("vs_cpu", {}).get("binary_flips") else 0


if __name__ == "__main__":
    sys.exit(main())
