"""Map a model set and cell size onto equal pixel blocks.

A model set is one distance network covering the whole cell, or several
networks ("tiles") that each cover a pixel rectangle, as in the Braille
spatial-split experiments. Every tile is evaluated on its own local
[0, 1] x [0, 2] grid, with supersampling chosen from the FULL cell width,
exactly as the reference worker calls `neuronte_coverage` per tile.

The NPU graph needs one tensor shape for all tiles, so each tile is cut into
blocks of a common size. Blocks never change which network a sample uses or
where the sample sits on that network's grid; they only group samples.
"""

from dataclasses import dataclass
from math import gcd
import json
from pathlib import Path

import numpy as np

from . import nrt

BRAILLE_FIRST = 0x2800


def auto_samples(cell_w):
    return 2 if cell_w < 24 else 1


@dataclass(frozen=True)
class Tile:
    net: nrt.Net
    mask: int | None  # Braille bit mask; None passes the scalar through unchanged
    x: int
    y: int
    w: int
    h: int
    name: str

    def scalar(self, scalar):
        if self.mask is None:
            return scalar
        return BRAILLE_FIRST | ((scalar - BRAILLE_FIRST) & self.mask)


@dataclass(frozen=True)
class Block:
    tile: int
    x: int  # pixel origin in the cell
    y: int


@dataclass(frozen=True)
class Layout:
    name: str
    cell_w: int
    cell_h: int
    samples: int
    tiles: tuple
    blocks: tuple
    block_w: int
    block_h: int

    @property
    def hidden(self):
        widths = {t.net.h for t in self.tiles}
        if len(widths) != 1:
            raise ValueError("grouped evaluation needs one hidden width for every tile")
        return widths.pop()

    @property
    def samples_per_block(self):
        return self.block_w * self.block_h * self.samples * self.samples

    @property
    def pixels_per_block(self):
        return self.block_w * self.block_h

    def sample_coords(self, block):
        """Local (tx, ty) for every sample of one block, in graph order.

        Order: pixel row, pixel column, sub-row, sub-column. The S*S
        subsamples of each pixel are consecutive, so a 1 x S*S average pool
        over the last axis yields the pixel's coverage.
        """
        tile = self.tiles[block.tile]
        s = self.samples
        sw, sh = tile.w * s, tile.h * s
        xs, ys = [], []
        for r in range(self.block_h):
            for c in range(self.block_w):
                for j in range(s):
                    for i in range(s):
                        col = (block.x - tile.x + c) * s + i
                        row = (block.y - tile.y + r) * s + j
                        xs.append((col + 0.5) / sw)
                        ys.append((row + 0.5) / sh * 2.0)
        return xs, ys

    def slope(self, tile):
        """`n->tau * (float)cell_w * (float)ss` for this tile, in float32."""
        t = tile.net
        return np.float32(np.float32(t.tau) * np.float32(tile.w)) * np.float32(self.samples)

    def describe(self):
        return {
            "name": self.name, "cell": [self.cell_w, self.cell_h], "samples": self.samples,
            "hidden": self.hidden, "tiles": len(self.tiles), "blocks": len(self.blocks),
            "block": [self.block_w, self.block_h],
            "samples_per_block": self.samples_per_block,
        }


def _blocks(tiles, cell_w, cell_h):
    bw, bh = cell_w, cell_h
    for t in tiles:
        for v in (t.x, t.w):
            bw = gcd(bw, v)
        for v in (t.y, t.h):
            bh = gcd(bh, v)
    owner = {}
    for index, t in enumerate(tiles):
        for y in range(t.y, t.y + t.h, bh):
            for x in range(t.x, t.x + t.w, bw):
                if (x, y) in owner:
                    raise ValueError("tiles overlap")
                owner[x, y] = index
    blocks = []
    for y in range(0, cell_h, bh):
        for x in range(0, cell_w, bw):
            if (x, y) not in owner:
                raise ValueError("tiles do not cover the cell")
            blocks.append(Block(owner[x, y], x, y))
    return tuple(blocks), bw, bh


def make_layout(name, tiles, cell_w, cell_h, samples=0):
    samples = samples or auto_samples(cell_w)
    blocks, bw, bh = _blocks(tiles, cell_w, cell_h)
    return Layout(name, cell_w, cell_h, samples, tuple(tiles), blocks, bw, bh)


BRAILLE_SPAN = (0x2800, 0x28FF)


def single(net_path, cell_w, cell_h, samples=0, name=None, span=BRAILLE_SPAN):
    """One network over the whole cell. A bundle is resolved to the member its
    range table routes `span` to; the meta still names the bundle file, so the
    CPU path runs the bundle's own router."""
    net = nrt.open_net(net_path, span)
    tile = Tile(net, None, 0, 0, cell_w, cell_h, Path(net_path).stem)
    return make_layout(name or Path(net_path).stem, [tile], cell_w, cell_h, samples)


def from_manifest(manifest_path, case_name, cell_w, cell_h, samples=0):
    """Build a layout from a kilix-bench Braille experiment manifest case."""
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())
    case = next((c for c in data["cases"] if c["name"] == case_name), None)
    if case is None:
        raise ValueError(f"no case {case_name!r} in the manifest")
    models = data["models"]
    if case.get("kind") != "parts":
        if len(case["members"]) != 1:
            raise ValueError("a bundle case must have one member")
        member = models[case["members"][0]]
        return single(manifest_path.parent / member["file"], cell_w, cell_h, samples, case_name)
    tiles = []
    for member, region in zip(case["members"], case["regions"], strict=True):
        x0, x1 = region["x_domain"]
        y0, y1 = region["y_domain"]
        edges = (x0 * cell_w, x1 * cell_w, y0 * cell_h / 2, y1 * cell_h / 2)
        if any(int(v) != v for v in edges):
            raise ValueError("cell dimensions must divide evenly into the layout")
        x, xe, y, ye = map(int, edges)
        net = nrt.open_net(manifest_path.parent / models[member]["file"])
        tiles.append(Tile(net, int(region["mask"]), x, y, xe - x, ye - y, member))
    return make_layout(case_name, tiles, cell_w, cell_h, samples)
