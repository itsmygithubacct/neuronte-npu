"""Exported graphs match the reference CPU library (ONNX Runtime, float32).

Needs onnxruntime, the Neuronte library and a Braille model manifest:

    NEURONTE_LIB=/path/libneuronte.so NEURONTE_MANIFEST=/path/manifest.json \
        PYTHONPATH=python python -m pytest tests
"""

import os

import pytest

ort = pytest.importorskip("onnxruntime")
LIB = os.environ.get("NEURONTE_LIB")
MANIFEST = os.environ.get("NEURONTE_MANIFEST")
pytestmark = pytest.mark.skipif(not (LIB and MANIFEST), reason="NEURONTE_LIB and NEURONTE_MANIFEST not set")


@pytest.fixture(scope="module")
def lib():
    from neuronte_npu import cpu
    return cpu.Library(LIB)


@pytest.mark.parametrize("case,cell,grouped", [
    ("braille48", "8x16", True),
    ("parts24-s0", "16x32", False),
    ("quadrants12-s1", "8x16", True),
    ("grid12-h4-s1", "16x32", True),
    ("grid12-h4-s1", "16x32", False),
])
def test_host_graph_matches_cpu(lib, case, cell, grouped):
    from neuronte_npu import check
    rec = check.check(MANIFEST, lib, case, cell, "host", grouped)
    assert rec["binary_flips"] == 0
    assert rec["max_abs"] < 1e-4


def test_bundle_routes_braille_to_one_member():
    from neuronte_npu import layout as L
    a = L.from_manifest(MANIFEST, "split-v5", 8, 16)
    b = L.from_manifest(MANIFEST, "braille48", 8, 16)
    assert (a.tiles[0].net.w2 == b.tiles[0].net.w2).all()
