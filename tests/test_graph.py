"""Graph variants compute the same coverage (ONNX Runtime, float32).

Needs onnx, onnxruntime and a Braille model manifest:

    NEURONTE_MANIFEST=/path/manifest.json PYTHONPATH=python python -m pytest tests
"""

import os

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
MANIFEST = os.environ.get("NEURONTE_MANIFEST")
pytestmark = pytest.mark.skipif(not MANIFEST, reason="NEURONTE_MANIFEST not set")
SCALARS = [0x2800 + i for i in range(0, 256, 5)]  # 52 scalars, every dot pattern class


def coverage(lay, grouped, pool):
    from neuronte_npu import graph
    model = graph.build(lay, len(SCALARS), grouped, "host", pool)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return graph.assemble(lay, sess.run(None, {"h1": graph.host_input(lay, SCALARS)})[0])


@pytest.mark.parametrize("case,cell,grouped", [
    ("braille48", "16x32", True),
    ("parts24-s0", "8x16", False),
    ("quadrants12-s1", "16x32", True),
])
@pytest.mark.parametrize("pool", ["conv", "host"])
def test_pool_forms_match_average_pool(case, cell, grouped, pool):
    from neuronte_npu import layout as L
    w, h = map(int, cell.split("x"))
    lay = L.from_manifest(MANIFEST, case, w, h)
    assert lay.samples > 1  # cells under 24 px wide average S*S samples per pixel
    ref, got = coverage(lay, grouped, "avg"), coverage(lay, grouped, pool)
    assert np.abs(got - ref).max() < 1e-6


def test_pool_host_output_is_per_sample():
    from neuronte_npu import graph, layout as L
    lay = L.from_manifest(MANIFEST, "braille48", 16, 32)
    out = graph.build(lay, 4, True, "host", "host").graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    assert dims == [1, len(lay.blocks), 4, lay.samples_per_block]
