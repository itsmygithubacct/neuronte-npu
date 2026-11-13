"""tools/compare.py on synthetic coverage dumps (numpy only)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("compare", Path(__file__).parents[1] / "tools" / "compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
W, H = 4, 8


def dump(tmp_path, name, a):
    p = tmp_path / name
    a.astype("<f4").tofile(p)
    return str(p)


def test_flips_and_exit_status(tmp_path):
    cpu = np.full((256, H, W), 0.2, np.float32)
    cpu[:, 0, 0] = 0.51
    got = cpu.copy()
    got[3, 0, 0] = 0.49           # one flip in one glyph
    got[7, 1, 1] = 0.25           # a difference that does not flip
    c, g = dump(tmp_path, "cpu.f32", cpu), dump(tmp_path, "got.f32", got)
    assert compare.vs_cpu(compare.load(g, W, H), compare.load(c, W, H)) == pytest.approx(
        {"max_abs": 0.05, "mean_abs": float(np.abs(got - cpu).mean()), "binary_flips": 1,
         "glyphs_with_flips": 1, "bit_identical": False})
    assert compare.main([f"{W}x{H}", g, "--cpu", c]) == 1
    assert compare.main([f"{W}x{H}", c, "--cpu", c]) == 0


def test_truth_metrics(tmp_path):
    truth = np.zeros((256, H, W), np.uint8)
    truth[:, :2, :2] = 1
    cov = truth.astype(np.float32)
    cov[0, 0, 0] = 0.4            # glyph 0 loses a pixel
    r = compare.vs_truth(cov, truth.astype(np.float32))
    assert (r["exact_glyphs"], r["wrong_pixels"]) == (255, 1)
    assert r["iou_min"] == pytest.approx(3 / 4)


def test_rejects_wrong_size(tmp_path):
    p = dump(tmp_path, "short.f32", np.zeros(10, np.float32))
    with pytest.raises(SystemExit):
        compare.load(p, W, H)
