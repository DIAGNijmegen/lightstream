import collections.abc as container_abcs
from itertools import repeat
from typing import NamedTuple, Union

from dataclasses import dataclass


B_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3


# inspired by torch/nn/modules/utils.py
def _ntuple(n):
    def parse(x, default=0):
        if isinstance(x, container_abcs.Iterable):
            if len(x) == n:
                return x
            elif len(x) == n - 1:
                return tuple([default, *x])
            else:
                return tuple(repeat(x[0], n))
        return tuple(repeat(x, n))

    return parse


# Utility named tuples, makes code more readable
class Sides(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int


@dataclass
class Box:
    y: int
    height: int
    x: int
    width: int
    sides: Union[Sides, None]


class IOShape(NamedTuple):
    batch: int
    channels: int
    height: int
    width: int


@dataclass
class Lost:
    top: int
    left: int
    bottom: int
    right: int

    def __str__(self):
        return "Lost(top:%2.1f, left:%2.1f, bottom:%2.1f, right:%2.1f)" % (self.top, self.left, self.bottom, self.right)


def _new_value_indices(data_shape, data_indices, old_value_indices):
    """
    This helper functions assumes we reconstruct feature maps and
    gradients in tiles from top-left to bottom-right. Using current tile
    index and old_value_indices it finds the relative indices of `data`
    which are unique for this tile (not earlier seen in other tiles).
    """
    rel_top, rel_bottom, rel_left, rel_right = 0, 0, 0, 0

    old_values_y = old_value_indices.y
    old_values_x = old_value_indices.x
    old_values_height = old_value_indices.height
    cur_y = data_indices.y
    cur_x = data_indices.x

    # Numerical drift safeguard:
    # very small forward jumps (typically 1 px) can happen in upstream
    # coordinate mapping for large streamed coordinates. We clamp those tiny
    # advances to the current cursor so overlap trimming remains strict and
    # we do not treat them as entirely new regions.
    if cur_x > old_values_x:
        drift_x = cur_x - old_values_x
        if drift_x <= 2:
            cur_x = old_values_x
    if cur_y > old_values_y:
        drift_y = cur_y - old_values_y
        if drift_y <= 2:
            cur_y = old_values_y

    # Check if new row
    if cur_x == 0:
        old_values_y = old_values_height
        old_values_height = cur_y + data_shape[H_DIM]
        old_values_x = 0

    # Check x-axis:
    # If this gradient is exactly on the border of old_value_indices
    # everything is new.
    if cur_x == old_values_x:
        rel_left = 0
        rel_right = data_shape[W_DIM]

    # If data_indices has some overlap with old_value_indices, trim unique
    # indices.
    else:
        assert old_values_x - cur_x >= 0, "Misses data in x-axis!"
        rel_left = old_values_x - cur_x
        rel_right = data_shape[W_DIM]

    # Check y-axis:
    # Equal to column logic (see above)
    if cur_y == old_values_y:
        rel_top = 0
        rel_bottom = data_shape[H_DIM]
    else:
        assert old_values_y - cur_y >= 0, "We miss data in y-axis"
        rel_top = old_values_y - cur_y
        rel_bottom = data_shape[H_DIM]

    # Update old-value-indices
    old_values_x += rel_right - rel_left

    assert rel_top >= 0, f"We miss data in y-axis before: {data_indices}"
    assert rel_left >= 0, f"We miss data in x-axis before: {data_indices}"

    new_value_indices = Box(rel_top, rel_bottom - rel_top, rel_left, rel_right - rel_left, None)
    old_value_indices = Box(int(old_values_y), int(old_values_height), int(old_values_x), 0, None)

    return new_value_indices, old_value_indices
