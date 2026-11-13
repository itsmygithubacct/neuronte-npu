"""tools/choose-forms.py on a synthetic run directory (numpy only)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("choose_forms", Path(__file__).parents[1] / "tools" / "choose-forms.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)
W, H = 4, 8


def run(d, kind, name, meta, wall_ns, cov, dtype=None, refine=0.0, rknn=""):
    rec = {"mode": kind, "backend": "cpu" if kind == "cpu" else "npu-rknn", "meta": meta, "rknn": rknn,
           "cell": [W, H], "glyphs_per_round": 256,
           "rounds": [{"warmup": False, "wall": wall_ns}] * 3}
    if kind == "npu":
        rec["info"] = {"dtype": dtype, "refine_margin": refine}
    (d / f"{name}.json").write_text(json.dumps(rec))
    cov.astype("<f4").tofile(d / f"{name}.f32")


def test_choose_and_assemble(tmp_path):
    runs, models, dest = tmp_path / "runs", tmp_path / "models", tmp_path / "set"
    runs.mkdir(), models.mkdir()
    cpu = np.full((256, H, W), 0.2, np.float32)
    cpu[:, 0, 0] = 0.51
    flipped = cpu.copy()
    flipped[5, 0, 0] = 0.49
    m16, m8 = "x-4x8-b64-g-host", "x-4x8-b64-g-host-w8a8-mmse"
    run(runs, "cpu", f"cpu-{m16}", f"models/{m16}.meta", 256_000_000, cpu)
    run(runs, "npu", f"npu-{m16}", f"models/{m16}.meta", 25_600_000, cpu + 0.01, "float16", rknn=f"m/{m16}.rknn")
    run(runs, "npu", f"npu-{m8}", f"models/{m8}.meta", 12_800_000, flipped, "int8", rknn=f"m/{m8}.rknn")
    run(runs, "npu", f"npu-{m16}-r0.05", f"models/{m16}.meta", 51_200_000, cpu, "float16", 0.05,
        rknn=f"m/{m16}.rknn")
    rows = cf.choose(str(runs))
    assert len(rows) == 1
    r = rows[0]
    assert r["config"] == "x-4x8"
    assert r["chosen"]["file"] == f"npu-{m16}.json"            # int8 is faster but flips a pixel
    assert abs(r["speedup"] - 10.0) < 1e-9
    assert {q["file"]: q.get("rejected") for q in r["candidates"]}[f"npu-{m8}.json"] == "1 flips"

    (models / f"{m16}.meta").write_text("neuronte-npu-meta 1\nhost-l1 a.l1 1 2 3\nrefine 0.3\n")
    (models / f"{m16}.rknn").write_bytes(b"rknn")
    (models / "a.l1").write_bytes(b"l1")
    r["chosen"]["refine"] = 0.05                                 # as if the refined run had won
    forms = cf.assemble(rows, str(runs), str(models), str(dest))
    assert forms[0]["meta"] == f"{m16}-r0.05.meta" and forms[0]["refine"] == 0.05
    meta = (dest / f"{m16}-r0.05.meta").read_text().splitlines()
    assert meta.count("refine 0.05") == 1 and "refine 0.3" not in meta
    sums = dict(line.split("  ")[::-1] for line in (dest / "SHA256SUMS").read_text().splitlines())
    assert set(sums) == {f"{m16}-r0.05.meta", f"{m16}-r0.05.rknn", "a.l1", "forms.json"}
    for name, digest in sums.items():
        assert hashlib.sha256((dest / name).read_bytes()).hexdigest() == digest


def test_no_exact_form(tmp_path):
    cpu = np.full((256, H, W), 0.2, np.float32)
    cpu[:, 0, 0] = 0.51
    bad = cpu.copy()
    bad[0, 0, 0] = 0.4
    run(tmp_path, "cpu", "cpu-y-4x8-b64-d-host", "m/y-4x8-b64-d-host.meta", 1000, cpu)
    run(tmp_path, "npu", "npu-y-4x8-b64-d-host", "m/y-4x8-b64-d-host.meta", 100, bad, "float16")
    rows = cf.choose(str(tmp_path))
    assert "chosen" not in rows[0]
    assert cf.main([str(tmp_path)]) == 1
