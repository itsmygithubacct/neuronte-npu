"""tools/summarize.py: the devfreq time-in-state delta from run-set.sh's .freq files."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("summarize", Path(__file__).parents[1] / "tools" / "summarize.py")
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)


def test_devfreq_delta(tmp_path):
    p = tmp_path / "npu-x.freq"
    p.write_text("before dmc 528000000 1000\nbefore dmc 2112000000 50\nbefore npu 1000000000 7\n"
                 "after dmc 528000000 1300\nafter dmc 2112000000 150\nafter npu 1000000000 7\n")
    d = summarize.devfreq(str(p))
    assert d["dmc"]["ms"] == 400
    assert abs(d["dmc"]["mean_mhz"] - (528 * 300 + 2112 * 100) / 400) < 1e-9
    assert d["dmc"]["at_max"] == 0.25
    assert "npu" not in d                         # no time accrued: no entry


def test_devfreq_missing(tmp_path):
    assert summarize.devfreq(str(tmp_path / "absent.freq")) is None
