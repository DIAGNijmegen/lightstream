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


def stitch_clipped(output: Any, tile: Any, destination_y: int, destination_x: int) -> None:
    """Stitch ``tile`` at an origin, clipping both windows to ``output``."""
    height, width = tile.shape[-2:]
    y0, x0 = max(0, destination_y), max(0, destination_x)
    y1 = min(output.shape[-2], destination_y + height)
    x1 = min(output.shape[-1], destination_x + width)
    if y1 <= y0 or x1 <= x0:
        return
    sy0, sx0 = y0 - destination_y, x0 - destination_x
    stitch_window(output, tile, (y0, y1, x0, x1),
                  (sy0, sy0 + y1 - y0, sx0, sx0 + x1 - x0))
