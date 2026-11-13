#!/usr/bin/env python3
"""Pick, per model configuration, the fastest NPU form whose binary output is
identical to the CPU library's, from one run-set.sh output directory.

    choose-forms.py OUTDIR [--json] [--assemble MODELDIR DEST [--nrt DIR]]

OUTDIR holds, per configuration (case and cell), a CPU run (cpu-STEM.json and
.f32) and any number of NPU runs of that configuration's models: fp16, int8,
with or without --refine (npu-STEM[-TAG].json and .f32). A run qualifies when
it ran on the NPU and its coverage has no binary flips against the CPU dump
(the same 0.5 rule as compare.py, over all 256 Braille scalars). Of the
qualifying runs the fastest (median µs per glyph) is chosen. A configuration
with no qualifying run is reported and left out.

--assemble copies each chosen model from MODELDIR into DEST: its .meta (with
a "refine MARGIN" line when the chosen run used refinement, so the backend
applies it by default; such a form is named STEM-rMARGIN), .rknn under the
same stem, and sidecars (.l1, .pos), and writes
DEST/forms.json (the choice and its measurements) and DEST/SHA256SUMS. With
--nrt DIR it also copies the .nrt networks the metas name, checking each
file's sha256 against the meta.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402

CONFIG = re.compile(r"^(?P<config>.+)-b\d+-[dg]-")


def config_of(meta_path):
    stem = os.path.basename(meta_path)[:-len(".meta")]
    m = CONFIG.match(stem)
    if not m:
        raise SystemExit(f"{meta_path}: cannot tell the configuration from the file name")
    return m.group("config"), stem


def median_wall(d):
    rounds = [r for r in d["rounds"] if not r["warmup"]]
    return st.median(r["wall"] for r in rounds) / d["glyphs_per_round"] / 1e3


def load_runs(outdir):
    cpu, npu = {}, {}
    for path in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        name = os.path.basename(path)
        if not name.startswith(("cpu-", "npu-")):
            continue
        d = json.load(open(path))
        if "rounds" not in d:
            print(f"skipping {name}: {d.get('error', 'no rounds')}", file=sys.stderr)
            continue
        config, stem = config_of(d["meta"])
        rec = {"file": name, "meta": d["meta"], "rknn": d.get("rknn", ""), "stem": stem, "cell": d["cell"],
               "wall": median_wall(d), "dump": path[:-len(".json")] + ".f32", "backend": d.get("backend")}
        if d["mode"] == "cpu":
            cpu[config] = rec
        else:
            info = d.get("info") or {}
            rec.update(dtype=info.get("dtype"), refine=info.get("refine_margin", 0.0))
            npu.setdefault(config, []).append(rec)
    return cpu, npu


def choose(outdir):
    cpu, npu = load_runs(outdir)
    rows = []
    for config in sorted(set(cpu) | set(npu)):
        c = cpu.get(config)
        if not c:
            rows.append({"config": config, "error": "no CPU run"})
            continue
        w, h = c["cell"]
        ref = compare.load(c["dump"], w, h)
        cands = []
        for r in npu.get(config, []):
            q = dict(r)
            if r["backend"] != "npu-rknn":
                q["rejected"] = f"backend {r['backend']}"
            else:
                v = compare.vs_cpu(compare.load(r["dump"], w, h), ref)
                q.update(flips=v["binary_flips"], max_abs=v["max_abs"])
                if v["binary_flips"]:
                    q["rejected"] = f"{v['binary_flips']} flips"
            cands.append(q)
        ok = sorted((q for q in cands if "rejected" not in q), key=lambda q: q["wall"])
        row = {"config": config, "cell": f"{w}x{h}", "cpu_wall": c["wall"], "candidates": cands}
        if ok:
            row["chosen"] = ok[0]
            row["speedup"] = c["wall"] / ok[0]["wall"]
        rows.append(row)
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble(rows, outdir, modeldir, dest, nrt_dir=None):
    os.makedirs(dest, exist_ok=True)
    files, forms = set(), []
    for row in rows:
        ch = row.get("chosen")
        if not ch:
            continue
        stem = ch["stem"]
        lines = open(os.path.join(modeldir, stem + ".meta")).read().splitlines()
        lines = [x for x in lines if not x.startswith("refine ")]
        if ch["refine"] and ch["refine"] > 0:
            lines.append(f"refine {ch['refine']:g}")
        # a model used both plainly and with refinement keeps one meta per form
        out_stem = stem + (f"-r{ch['refine']:g}" if ch["refine"] and ch["refine"] > 0 else "")
        with open(os.path.join(dest, out_stem + ".meta"), "w") as f:
            f.write("\n".join(lines) + "\n")
        files.add(out_stem + ".meta")
        rknn = out_stem + ".rknn"  # every form is STEM.meta + STEM.rknn, as run-set.sh expects
        shutil.copyfile(os.path.join(modeldir, os.path.basename(ch["rknn"])), os.path.join(dest, rknn))
        files.add(rknn)
        for x in lines:
            k = x.split()
            if k and k[0] in ("host-l1", "pos"):
                shutil.copyfile(os.path.join(modeldir, k[1]), os.path.join(dest, k[1]))
                files.add(k[1])
            if nrt_dir and k and k[0] == "tile":
                src, want = os.path.join(nrt_dir, k[7]), k[8]
                if sha256(src) != want:
                    raise SystemExit(f"{src}: sha256 differs from {out_stem}.meta")
                shutil.copyfile(src, os.path.join(dest, k[7]))
                files.add(k[7])
        forms.append({"config": row["config"], "cell": row["cell"], "meta": out_stem + ".meta", "rknn": rknn,
                      "dtype": ch["dtype"], "refine": ch["refine"], "npu_us_per_glyph": round(ch["wall"], 2),
                      "cpu_us_per_glyph": round(row["cpu_wall"], 2), "speedup": round(row["speedup"], 2),
                      "max_abs_vs_cpu": ch["max_abs"], "binary_flips_vs_cpu": ch["flips"]})
    with open(os.path.join(dest, "forms.json"), "w") as f:
        json.dump({"runs": os.path.basename(os.path.normpath(outdir)), "forms": forms}, f, indent=1)
    files.add("forms.json")
    with open(os.path.join(dest, "SHA256SUMS"), "w") as f:
        for name in sorted(files):
            f.write(f"{sha256(os.path.join(dest, name))}  {name}\n")
    return forms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--assemble", nargs=2, metavar=("MODELDIR", "DEST"))
    ap.add_argument("--nrt", metavar="DIR")
    a = ap.parse_args(argv)
    rows = choose(a.outdir)
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            if "chosen" in r:
                ch = r["chosen"]
                form = ch["dtype"] + (f"+refine {ch['refine']:g}" if ch["refine"] and ch["refine"] > 0 else "")
                print(f"{r['config']:28s} {form:20s} {ch['wall']:8.1f} µs/glyph  CPU {r['cpu_wall']:8.1f}  "
                      f"{r['speedup']:5.1f}x  ({ch['file']})")
            else:
                why = r.get("error") or "; ".join(f"{q['file']}: {q['rejected']}" for q in r["candidates"]) \
                    or "no NPU run"
                print(f"{r['config']:28s} NO EXACT FORM: {why}")
    if a.assemble:
        forms = assemble(rows, a.outdir, a.assemble[0], a.assemble[1], a.nrt)
        print(f"assembled {len(forms)} configurations into {a.assemble[1]}", file=sys.stderr)
    return 0 if all("chosen" in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
