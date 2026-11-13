"""Read Neuronte version 1 distance networks (alone, or routed from a bundle).

The layout matches the reference C loader (`net_open` in neuronte.c): a
204-byte header, then the four affine layers as little-endian float32.
"""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

OCT = 6
POS = 2 + 4 * OCT  # 26 position features
CODE = 64  # 21 scalar bits plus 43 parity bits
NMASK = CODE - 21
IN = POS + CODE
LAYERS = 3
HEADER = 8 + 5 * 4 + 4 + NMASK * 4
MAX_H = 1024


@dataclass(frozen=True)
class Net:
    """One 90 -> H -> H -> H -> 1 distance network."""

    h: int
    tau: float
    masks: tuple  # 43 parity masks
    w1: np.ndarray  # (h, 90), output-major as stored
    b1: np.ndarray  # (h,)
    w2: np.ndarray  # (h, h), [out, in]
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    w4: np.ndarray  # (h,)
    b4: np.float32

    @property
    def w1_pos(self):
        return self.w1[:, :POS]

    @property
    def w1_code(self):
        return self.w1[:, POS:]


def params(h):
    return h * IN + h + 2 * (h * h + h) + h + 1


def file_len(h):
    return HEADER + 4 * params(h) if 1 <= h <= MAX_H else 0


def read_net(data):
    """Parse one version 1 image; raise ValueError on any mismatch."""
    if len(data) < 28 or data[:8] != b"NEURONTE":
        raise ValueError("not a Neuronte image")
    version, octaves, code, h, layers = struct.unpack_from("<5I", data, 8)
    if version != 1:
        raise ValueError(f"expected a version 1 distance network, found version {version}")
    if octaves != OCT or code != CODE or layers != LAYERS or not file_len(h):
        raise ValueError("unsupported distance network shape")
    if len(data) != file_len(h):
        raise ValueError("distance network length does not match its width")
    (tau,) = struct.unpack_from("<f", data, 28)
    masks = struct.unpack_from(f"<{NMASK}I", data, 32)
    if not np.isfinite(tau) or tau <= 0 or any(m == 0 or m >= 1 << 21 for m in masks):
        raise ValueError("invalid tau or parity mask")
    floats = np.frombuffer(data, dtype="<f4", offset=HEADER).astype(np.float32)
    if not np.isfinite(floats).all():
        raise ValueError("non-finite weight")
    off = 0

    def take(n, shape=None):
        nonlocal off
        part = floats[off:off + n]
        off += n
        return part.reshape(shape) if shape else part

    w1 = take(h * IN, (h, IN))
    b1 = take(h)
    w2 = take(h * h, (h, h))
    b2 = take(h)
    w3 = take(h * h, (h, h))
    b3 = take(h)
    w4 = take(h)
    b4 = np.float32(take(1)[0])
    assert off == floats.size
    return Net(h, float(np.float32(tau)), tuple(masks), w1, b1, w2, b2, w3, b3, w4, b4)


BUNDLE_HEAD = 28


def bundle_member(data, first, last):
    """The version 1 member of a version 2 bundle that draws every scalar in
    [first, last], from the bundle's range table. Raise ValueError when the
    range table does not send the whole span to one member (the gate and the
    fallback net are not evaluated here)."""
    if data[:8] != b"NEURONTE" or struct.unpack_from("<I", data, 8)[0] != 2:
        raise ValueError("not a version 2 bundle")
    k, r, _fallback, g = struct.unpack_from("<4I", data, 12)
    lens = struct.unpack_from(f"<{k}I", data, BUNDLE_HEAD)
    off = BUNDLE_HEAD + 4 * k
    ranges = [struct.unpack_from("<3I", data, off + 12 * i) for i in range(r)]
    off += 12 * r
    if g:
        off += 4 * (NMASK + g * CODE + g + k * g + k)
    hit = [net for lo, hi, net in ranges if lo <= first and last <= hi]
    if len(hit) != 1:
        raise ValueError(f"the bundle's range table does not route {first:#x}..{last:#x} to one member")
    for i in range(hit[0]):
        off += lens[i]
    return data[off:off + lens[hit[0]]]


def open_net(path, span=None):
    """Open a version 1 net, or with `span` = (first, last) the member of a
    version 2 bundle that draws that scalar span."""
    data = Path(path).read_bytes()
    if span is not None and data[8:12] == struct.pack("<I", 2):
        data = bundle_member(data, *span)
    return read_net(data)


def parity(v):
    return bin(v).count("1") & 1


def code_features(scalar, masks):
    """The 64 character features of `code_of`, each -1.0 or +1.0."""
    bits = [(scalar >> i) & 1 for i in range(21)]
    bits += [parity(scalar & m) for m in masks]
    return np.where(np.asarray(bits) != 0, np.float32(1), np.float32(-1)).astype(np.float32)


def axis_part(net, axis, t):
    """One axis's columns of layer 1 at coordinate t, accumulated like `axis_part`.

    Axis 0 is x in [0, 1], axis 1 is y in [0, 2]. The C code evaluates the
    sine and cosine in double precision, rounds them to float, and sums in
    float order: the linear term first, then sin + cos for each octave.
    """
    w = net.w1_pos
    v = np.float32(2.0 * t - 1.0) if axis == 0 else np.float32(t - 1.0)
    out = (v * w[:, axis]).astype(np.float32)
    for k in range(OCT):
        omega = 2.0 * np.pi * float(1 << k)
        sv = np.float32(np.sin(omega * t))
        cv = np.float32(np.cos(omega * t))
        out = (out + (sv * w[:, 2 + 4 * k + axis] + cv * w[:, 4 + 4 * k + axis]).astype(np.float32)).astype(np.float32)
    return out


def char_part(net, scalar):
    """Layer-1 bias plus the character columns, accumulated like `char_part`."""
    c = net.b1.copy()
    feats = code_features(scalar, net.masks)
    for i in range(CODE):
        c = (c + feats[i] * net.w1_code[:, i]).astype(np.float32)
    return c
