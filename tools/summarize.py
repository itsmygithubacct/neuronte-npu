#!/usr/bin/env python3
"""Summarise neuronte_npu_bench JSON files: one line each, medians over the
measured rounds (warmup rounds excluded), in microseconds per glyph.

    summarize.py FILE.json ...        text table
    summarize.py --json FILE.json ... one JSON record per file

When run-set.sh's FILE.freq is next to FILE.json, the record also has the
time-weighted mean clock of every devfreq device over the run (open, warmup
and measured rounds together) and the share of that time at its top
frequency; the text table shows the DDR controller's (dmc).
"""
import json
import statistics as st
import sys


def devfreq(path):
    """{device: {ms, mean_mhz, at_max}} from a .freq file, or None without one."""
    try:
        text = open(path).read()
    except OSError:
        return None
    acc = {}
    for line in text.splitlines():
        when, dev, hz, ms = line.split()
        acc.setdefault(dev, {}).setdefault(int(hz), [0, 0])[when == "after"] = int(ms)
    out = {}
    for dev, states in acc.items():
        spent = {hz: after - before for hz, (before, after) in states.items()}
        total = sum(spent.values())
        if total > 0:
            out[dev] = {"ms": total, "mean_mhz": sum(hz * ms for hz, ms in spent.items()) / total / 1e6,
                        "at_max": spent[max(spent)] / total}
    return out


def summary(path):
    d = json.load(open(path))
    rec = {"file": path.rsplit("/", 1)[-1], "mode": d.get("mode"), "backend": d.get("backend")}
    if "rounds" not in d:
        rec["error"] = d.get("error", "no rounds")
        return rec
    rounds = [r for r in d["rounds"] if not r["warmup"]]
    g = d["glyphs_per_round"]
    med = lambda f: st.median(f(r) for r in rounds) / g / 1e3
    rec.update(batch=d["batch"], cell=d["cell"], rounds=len(rounds), glyphs_per_round=g,
               wall=med(lambda r: r["wall"]), cpu=med(lambda r: r["cpu"]),
               wall_min=min(r["wall"] for r in rounds) / g / 1e3, wall_max=max(r["wall"] for r in rounds) / g / 1e3,
               **{k: med(lambda r, k=k: r["stages"][k]) for k in ("features", "input", "run", "output", "assemble",
                                                                    "cpu_eval")},
               open_ms=d["open_ns"] / 1e6, first_ms=d["first"]["wall"] / 1e6, info=d.get("info"),
               npu_temp=d["hardware_end"]["thermal"].get("npu-thermal"),
               load_start=d["hardware_start"]["loadavg"].split()[0])
    if d.get("single"):
        lat = [s["wall"] for s in d["single"]] if isinstance(d["single"][0], dict) else d["single"]
        rec["single_us"] = st.median(lat) / 1e3
    freq = devfreq(path[:-5] + ".freq") if path.endswith(".json") else None
    if freq is not None:
        rec["devfreq"] = freq
    return rec


def main(argv):
    as_json = argv[:1] == ["--json"]
    files = argv[1:] if as_json else argv
    for f in files:
        r = summary(f)
        if as_json:
            print(json.dumps(r))
        elif "error" in r:
            print(f"{r['file']:52s} ERROR {r['error']}")
        else:
            info = r["info"] or {}
            print(f"{r['file']:52s} {r['backend']:12s} b{r['batch']:<4d} wall {r['wall']:8.1f} "
                  f"[{r['wall_min']:.1f}-{r['wall_max']:.1f}] cpu {r['cpu']:7.1f} | fill {r['features']:6.1f} "
                  f"run {r['run']:7.1f} asm {r['assemble']:5.1f} eval {r['cpu_eval']:8.1f} | open {r['open_ms']:6.1f} ms"
                  f" first {r['first_ms']:7.1f} ms pipe {info.get('pipeline', '-')}"
                  + (f" single {r['single_us']:8.1f} us" if "single_us" in r else "")
                  + (f" ddr {r['devfreq']['dmc']['mean_mhz']:.0f} MHz" if "dmc" in r.get("devfreq", {}) else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
