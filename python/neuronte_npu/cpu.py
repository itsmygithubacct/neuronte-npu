"""The reference CPU path, through the Neuronte C library (ctypes).

Tiles are evaluated the way the kilix-bench split worker does it: one
`neuronte_coverage` call per tile with the tile's masked scalar, the tile's
pixel size and the supersampling of the FULL cell, then copied into place.
"""

import ctypes
import numpy as np


class Library:
    def __init__(self, path):
        lib = ctypes.CDLL(str(path))
        lib.neuronte_open_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.neuronte_open_file.restype = ctypes.c_int
        lib.neuronte_coverage.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        lib.neuronte_coverage.restype = ctypes.c_int
        lib.neuronte_close.argtypes = [ctypes.c_void_p]
        lib.neuronte_close.restype = None
        self.lib = lib

    def coverage(self, layout, net_paths, scalars):
        """[B, cell_h, cell_w] coverage for each scalar, by the reference library."""
        handles = []
        try:
            for path in net_paths:
                h = ctypes.c_void_p()
                rc = self.lib.neuronte_open_file(str(path).encode(), ctypes.byref(h))
                if rc:
                    raise RuntimeError(f"neuronte_open_file({path}) failed: {rc}")
                handles.append(h)
            out = np.zeros((len(scalars), layout.cell_h, layout.cell_w), dtype=np.float32)
            for b, s in enumerate(scalars):
                for tile, handle in zip(layout.tiles, handles, strict=True):
                    buf = np.zeros(tile.w * tile.h, dtype=np.float32)
                    rc = self.lib.neuronte_coverage(handle, tile.scalar(s), tile.w, tile.h, layout.samples,
                                                    buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                    if rc:
                        raise RuntimeError(f"neuronte_coverage failed: {rc}")
                    out[b, tile.y:tile.y + tile.h, tile.x:tile.x + tile.w] = buf.reshape(tile.h, tile.w)
            return out
        finally:
            for h in handles:
                self.lib.neuronte_close(h)
