"""Build the static ONNX graph that evaluates a model set on an NPU.

Tensor layout is [1, channels, glyphs, samples]: the glyph batch sits on the
height axis and each block's samples on the width axis, so every layer is a
1 x 1 convolution over one large plane and a whole batch is one launch.

    code   [1, T*64, B, 1]   character features of each glyph, per tile (+-1)
    conv   -> c [1, G*H, B, 1]          layer 1, character columns + bias
    add    c + P[1, G*H, 1, M] -> ReLU  layer 1, constant position columns
    conv   grouped (G) -> ReLU          layer 2
    conv   grouped (G) -> ReLU          layer 3
    conv   grouped (G)                  layer 4, with 0.5 + slope * f folded in
    clip   [0, 1]
    pool   average over each pixel's S*S consecutive samples
    cover  [1, G, B, bh*bw]             per-block pixel coverage

The CPU decomposition (character part once per glyph, position parts once
per axis) survives: the character part is a per-glyph tensor and the
position part is a baked constant, added by broadcasting.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import nrt

OPSET = 13


def position_constant(layout):
    """P[g*H + o, m] = x part + y part of block g's network at sample m."""
    h = layout.hidden
    m = layout.samples_per_block
    p = np.zeros((len(layout.blocks) * h, m), dtype=np.float32)
    cache = {}
    for g, block in enumerate(layout.blocks):
        net = layout.tiles[block.tile].net
        xs, ys = layout.sample_coords(block)
        for k, (tx, ty) in enumerate(zip(xs, ys)):
            kx, ky = (block.tile, 0, tx), (block.tile, 1, ty)
            if kx not in cache:
                cache[kx] = nrt.axis_part(net, 0, tx)
            if ky not in cache:
                cache[ky] = nrt.axis_part(net, 1, ty)
            p[g * h:(g + 1) * h, k] = (cache[kx] + cache[ky]).astype(np.float32)
    return p


def weights(layout, grouped=True):
    """All graph constants, as float32 arrays."""
    h = layout.hidden
    g_count, t_count = len(layout.blocks), len(layout.tiles)
    code_w = np.zeros((g_count * h, t_count * nrt.CODE, 1, 1), dtype=np.float32)
    code_b = np.zeros(g_count * h, dtype=np.float32)
    rows = g_count * h
    cols = h if grouped else g_count * h
    w2 = np.zeros((rows, cols, 1, 1), dtype=np.float32)
    w3 = np.zeros((rows, cols, 1, 1), dtype=np.float32)
    b2 = np.zeros(rows, dtype=np.float32)
    b3 = np.zeros(rows, dtype=np.float32)
    w4 = np.zeros((g_count, cols, 1, 1), dtype=np.float32)
    b4 = np.zeros(g_count, dtype=np.float32)
    for g, block in enumerate(layout.blocks):
        tile = layout.tiles[block.tile]
        net, t = tile.net, block.tile
        sl = slice(g * h, (g + 1) * h)
        code_w[sl, t * nrt.CODE:(t + 1) * nrt.CODE, 0, 0] = net.w1_code
        code_b[sl] = net.b1
        inner = slice(0, h) if grouped else sl
        w2[sl, inner, 0, 0] = net.w2
        w3[sl, inner, 0, 0] = net.w3
        b2[sl], b3[sl] = net.b2, net.b3
        slope = layout.slope(tile)
        w4[g, inner, 0, 0] = (slope * net.w4).astype(np.float32)
        b4[g] = np.float32(np.float32(0.5) + slope * net.b4)
    return {"code_w": code_w, "code_b": code_b, "pos": position_constant(layout)[None, :, None, :],
            "w2": w2, "b2": b2, "w3": w3, "b3": b3, "w4": w4, "b4": b4}


L1_ACTIVATIONS = ("host", "input", "input-relu", "input-convadd", "input-shared", "tiled", "deconv", "relu", "conv",
                  "clip")
POS_INPUTS = ("input", "input-relu", "input-convadd", "input-shared")
POOLS = ("avg", "conv", "host")


def build(layout, batch, grouped=True, l1="host", pool="avg"):
    """Return an onnx.ModelProto for `batch` glyphs of this layout.

    `pool` chooses how each pixel's S*S sample coverages are averaged:
      avg   AveragePool [1, S*S] (default).
      conv  a depthwise Conv with a [1, S*S] kernel of 1/(S*S) and the same
            stride: the same arithmetic on the convolution engine.
      host  no pooling: the output is per sample, [1, G, B, M], and the
            host averages. Only for the C backend's "pool host" meta line.

    `l1` chooses how layer 1 combines the per-glyph and per-sample parts:
      host    layer 1 runs on the host: the graph INPUT is h1 [1, C, B, M],
            ReLU(character part + position part), which the backend writes
            straight into the NPU's native buffer. The NPU runs layers 2-4,
            which hold all but about 1% of the multiply-adds. Default; see
            host_layer1() for the data the backend needs.
      input   the position term is a second graph INPUT, pos [1, C, B, M],
            which the backend fills once at open (P tiled over the batch).
            Add(c, pos) broadcasts only c, along samples; an identity
            depthwise conv carries the ReLU. Correct on the NPU, but the
            Add and the identity conv are two extra full-size passes.
      input-relu  as input, with a plain Relu after the Add.
      input-convadd  as input, but the identity conv runs on pos and the
            character part is added to its output: Relu(Conv(pos) + c), which
            a compiler can fuse into one ConvAddRelu pass.
      input-shared  as input, but pos is [1, C, 1, M] (not tiled over the
            batch) and Add broadcasts it along the glyph axis.
      tiled   Add(c [1,C,B,1], P tiled over the batch [1,C,B,M]): only the
            runtime tensor is broadcast (along samples), the constant is
            full size. Then an identity depthwise conv carries the ReLU.
            The compiler folds the tiled constant back into a broadcast, so
            this computes the same wrong values as relu/conv/clip.
      deconv  no broadcasting. The character conv emits channel pairs
            (c_j, 1); a grouped ConvTranspose with a 1 x M kernel and stride
            multiplies them by (ones, P_j), giving c_j + P_j at every sample.
            Correct on RKNPU driver 0.9.3 but very slow: the compiler lowers
            the long kernel into huge buffers.
    The other three keep a broadcast Add(c, P), which that driver computes
    WRONGLY (wrong position term in most channel blocks), and differ only in
    the activation that follows it:
      relu  Add -> Relu. Compilers fuse this into one AddRelu kernel, which
            fails to submit (times out) on RKNPU driver 0.9.3.
      conv  Add -> identity depthwise 1x1 Conv -> Relu. Same values; the
            activation rides on a convolution, which that driver runs.
      clip  Add -> Clip(0, +inf).
    """
    if l1 not in L1_ACTIVATIONS:
        raise ValueError(f"l1 must be one of {L1_ACTIVATIONS}")
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}")
    h = layout.hidden
    g_count, t_count = len(layout.blocks), len(layout.tiles)
    m = layout.samples_per_block
    s2 = layout.samples * layout.samples
    groups = g_count if grouped else 1
    consts = weights(layout, grouped)
    init = [numpy_helper.from_array(v, k) for k, v in consts.items()]
    zero, one = np.array(0.0, np.float32), np.array(1.0, np.float32)
    init += [numpy_helper.from_array(zero, "lo"), numpy_helper.from_array(one, "hi")]
    rows = g_count * h
    if l1 == "host":
        init = [i for i in init if i.name not in ("code_w", "code_b", "pos")]
        nodes = []
    elif l1 == "deconv":
        pair_w = np.zeros((2 * rows, t_count * nrt.CODE, 1, 1), dtype=np.float32)
        pair_w[0::2] = consts["code_w"]
        pair_b = np.zeros(2 * rows, dtype=np.float32)
        pair_b[0::2] = consts["code_b"]
        pair_b[1::2] = 1.0
        spread = np.empty((2 * rows, 1, 1, m), dtype=np.float32)
        spread[0::2, 0, 0, :] = 1.0
        spread[1::2, 0, 0, :] = consts["pos"][0, :, 0, :]
        init = [i for i in init if i.name not in ("code_w", "code_b", "pos")]
        init += [numpy_helper.from_array(pair_w, "pair_w"), numpy_helper.from_array(pair_b, "pair_b"),
                 numpy_helper.from_array(spread, "spread")]
        nodes = [
            helper.make_node("Conv", ["code", "pair_w", "pair_b"], ["c2"], kernel_shape=[1, 1]),
            helper.make_node("ConvTranspose", ["c2", "spread"], ["a1"], kernel_shape=[1, m], strides=[1, m],
                             group=rows),
            helper.make_node("Relu", ["a1"], ["h1"]),
        ]
    else:
        if l1 in POS_INPUTS:
            init = [i for i in init if i.name != "pos"]
        elif l1 == "tiled":
            init = [i for i in init if i.name != "pos"]
            init.append(numpy_helper.from_array(np.ascontiguousarray(np.broadcast_to(
                consts["pos"], (1, rows, batch, m))), "pos"))
        nodes = [helper.make_node("Conv", ["code", "code_w", "code_b"], ["c"], kernel_shape=[1, 1])]
        if l1 == "input-convadd":
            init.append(numpy_helper.from_array(np.ones((rows, 1, 1, 1), np.float32), "eye_w"))
            init.append(numpy_helper.from_array(np.zeros(rows, np.float32), "eye_b"))
            nodes.append(helper.make_node("Conv", ["pos", "eye_w", "eye_b"], ["e0"], kernel_shape=[1, 1],
                                          group=rows))
            nodes.append(helper.make_node("Add", ["e0", "c"], ["a1"]))
        else:
            nodes.append(helper.make_node("Add", ["c", "pos"], ["a1"]))
    if l1 in ("deconv", "host"):
        pass
    elif l1 in ("relu", "input-relu", "input-convadd"):
        nodes.append(helper.make_node("Relu", ["a1"], ["h1"]))
    elif l1 == "clip":
        init.append(numpy_helper.from_array(np.array(np.finfo(np.float32).max, np.float32), "big"))
        nodes.append(helper.make_node("Clip", ["a1", "lo", "big"], ["h1"]))
    else:  # conv, tiled, input, input-shared
        init.append(numpy_helper.from_array(np.ones((rows, 1, 1, 1), np.float32), "eye_w"))
        init.append(numpy_helper.from_array(np.zeros(rows, np.float32), "eye_b"))
        nodes.append(helper.make_node("Conv", ["a1", "eye_w", "eye_b"], ["e1"], kernel_shape=[1, 1], group=rows))
        nodes.append(helper.make_node("Relu", ["e1"], ["h1"]))
    nodes += [
        helper.make_node("Conv", ["h1", "w2", "b2"], ["a2"], kernel_shape=[1, 1], group=groups),
        helper.make_node("Relu", ["a2"], ["h2"]),
        helper.make_node("Conv", ["h2", "w3", "b3"], ["a3"], kernel_shape=[1, 1], group=groups),
        helper.make_node("Relu", ["a3"], ["h3"]),
        helper.make_node("Conv", ["h3", "w4", "b4"], ["f"], kernel_shape=[1, 1], group=groups),
        helper.make_node("Clip", ["f", "lo", "hi"], ["v"]),
    ]
    out = "cover"
    pooled = s2 > 1 and pool != "host"
    if pooled and pool == "conv":
        init.append(numpy_helper.from_array(np.full((g_count, 1, 1, s2), 1.0 / s2, np.float32), "pool_w"))
        nodes.append(helper.make_node("Conv", ["v", "pool_w"], [out], kernel_shape=[1, s2], strides=[1, s2],
                                      group=g_count))
    elif pooled:
        nodes.append(helper.make_node("AveragePool", ["v"], [out], kernel_shape=[1, s2], strides=[1, s2]))
    else:
        nodes.append(helper.make_node("Identity", ["v"], [out]))
    if l1 == "host":
        inputs = [helper.make_tensor_value_info("h1", TensorProto.FLOAT, [1, rows, batch, m])]
    else:
        inputs = [helper.make_tensor_value_info("code", TensorProto.FLOAT, [1, t_count * nrt.CODE, batch, 1])]
    if l1 in POS_INPUTS:
        pos_batch = 1 if l1 == "input-shared" else batch
        inputs.append(helper.make_tensor_value_info("pos", TensorProto.FLOAT, [1, rows, pos_batch, m]))
    outputs = [helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, g_count, batch, m // s2 if pooled else m])]
    graph = helper.make_graph(nodes, f"neuronte_{layout.name}", inputs, outputs, init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="neuronte-npu")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def host_layer1(layout):
    """What the backend needs to compute h1 for the host graph, as float32:
    code_w [G*H, T*64], code_b [G*H], P [G*H, M]."""
    c = weights(layout)
    rows = c["code_b"].shape[0]
    return c["code_w"].reshape(rows, -1), c["code_b"], c["pos"].reshape(rows, -1)


def host_input(layout, scalars):
    """The h1 input of the host graph for a batch of scalars (reference)."""
    w, b, p = host_layer1(layout)
    code = code_input(layout, scalars)[0, :, :, 0]  # [T*64, B]
    c = w @ code + b[:, None]                        # [G*H, B]
    return np.maximum(c[None, :, :, None] + p[None, :, None, :], 0).astype(np.float32)


def position_input(layout, batch):
    """The `pos` input of the POS_INPUTS graphs: P tiled over `batch` glyphs
    (1 for input-shared)."""
    p = position_constant(layout)
    return np.ascontiguousarray(np.broadcast_to(p[None, :, None, :], (1, p.shape[0], batch, p.shape[1])))


def code_input(layout, scalars):
    """The graph input for a batch of scalars: [1, T*64, B, 1] float32."""
    t_count = len(layout.tiles)
    x = np.empty((1, t_count * nrt.CODE, len(scalars), 1), dtype=np.float32)
    for b, s in enumerate(scalars):
        for t, tile in enumerate(layout.tiles):
            x[0, t * nrt.CODE:(t + 1) * nrt.CODE, b, 0] = nrt.code_features(tile.scalar(s), tile.net.masks)
    return x


def assemble(layout, cover):
    """[1, G, B, bh*bw] block coverage -> [B, cell_h, cell_w] glyph masks.

    Per-sample output ([1, G, B, M], pool="host") is averaged first."""
    s2 = layout.samples * layout.samples
    if s2 > 1 and cover.shape[3] == layout.samples_per_block:
        cover = cover.reshape(*cover.shape[:3], -1, s2).mean(axis=4)
    _, g_count, batch, _ = cover.shape
    out = np.empty((batch, layout.cell_h, layout.cell_w), dtype=np.float32)
    bw, bh = layout.block_w, layout.block_h
    for g, block in enumerate(layout.blocks):
        out[:, block.y:block.y + bh, block.x:block.x + bw] = cover[0, g].reshape(batch, bh, bw)
    return out
