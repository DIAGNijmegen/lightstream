"""Pure geometry used by streaming setup and execution.

This module intentionally has no dependency on a model or on PyTorch.  Keeping
these calculations here makes the tiling contract independently testable.
"""

import math
from dataclasses import dataclass
from typing import Iterator, NamedTuple

B_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3


class Sides(NamedTuple):
    left: bool
    top: bool
    right: bool
    bottom: bool


@dataclass
class Box:
    y: int
    height: int
    x: int
    width: int
    sides: Sides | None = None


@dataclass
class Lost:
    top: int
    left: int
    bottom: int
    right: int

    def __str__(self) -> str:
        return "Lost(top:%2.1f, left:%2.1f, bottom:%2.1f, right:%2.1f)" % tuple(self.__dict__.values())


def tile_grid(image_height: int, image_width: int, tile_height: int, tile_width: int,
              step_height: int, step_width: int) -> tuple[int, int]:
    """Return the number of tile rows and columns needed to cover an image."""
    if step_height <= 0 or step_width <= 0:
        raise ValueError("tile steps must be positive")
    rows = math.ceil(max(1, image_height - tile_height) / step_height) + 1
    cols = math.ceil(max(1, image_width - tile_width) / step_width) + 1
    return (1 if image_height <= tile_height else rows, 1 if image_width <= tile_width else cols)


def iter_tiles(image_height: int, image_width: int, tile_height: int, tile_width: int,
               step_height: int, step_width: int, rows: int | None = None,
               cols: int | None = None) -> Iterator[tuple[int, int, Sides]]:
    """Yield clamped tile origins and edge markers in row-major order."""
    rows, cols = tile_grid(image_height, image_width, tile_height, tile_width,
                           step_height, step_width) if rows is None or cols is None else (rows, cols)
    for row in range(rows):
        for col in range(cols):
            y, x = row * step_height, col * step_width
            sides = Sides(col == 0, row == 0, x + tile_width >= image_width,
                          y + tile_height >= image_height)
            if sides.bottom:
                y = max(image_height - tile_height, 0)
            if sides.right:
                x = max(image_width - tile_width, 0)
            yield int(y), int(x), sides


def aligned_step(candidates: list[int] | tuple[int, ...], alignments: list[int] | tuple[int, ...]) -> int:
    """Choose the largest aligned step no greater than the safest candidate."""
    if not candidates:
        raise ValueError("at least one step candidate is required")
    alignment = 1
    for value in alignments:
        alignment = math.lcm(alignment, max(1, int(value)))
    return max(alignment, (min(map(int, candidates)) // alignment) * alignment)


def full_output_size(image_size: int, tile_size: int, tile_output_size: int, stride: int) -> int:
    """Calculate one dimension of the fully stitched output."""
    return (image_size - tile_size) // int(stride) + tile_output_size


def output_window(tile_origin: int, stride: int, output_size: int, tile_output_size: int,
                  lost_before: int, lost_after: int, at_start: bool, at_end: bool) -> tuple[int, int, int, int]:
    """Return destination start/end and source trim start/end for one axis."""
    before = 0 if at_start else lost_before
    after = 0 if at_end else lost_after
    source_start, source_end = before, tile_output_size - after
    destination_start = tile_origin // int(stride) + source_start
    destination_end = min(output_size, destination_start + source_end - source_start)
    return destination_start, destination_end, source_start, source_start + destination_end - destination_start
