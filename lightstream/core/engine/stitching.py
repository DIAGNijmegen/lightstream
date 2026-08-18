"""Tensor-agnostic helpers for stitching tile payloads into outputs."""

from typing import Any


def stitch_window(output: Any, tile: Any, destination: tuple[int, int, int, int],
                  source: tuple[int, int, int, int]) -> None:
    """Copy a spatial source window into a destination window in-place.

    ``output`` and ``tile`` only need NumPy/PyTorch-style four-dimensional
    slicing, which keeps this primitive independently testable.
    """
    dy0, dy1, dx0, dx1 = destination
    sy0, sy1, sx0, sx1 = source
    if (dy1 - dy0, dx1 - dx0) != (sy1 - sy0, sx1 - sx0):
        raise ValueError("source and destination windows must have equal spatial shape")
    output[:, :, dy0:dy1, dx0:dx1] = tile[:, :, sy0:sy1, sx0:sx1]
