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

    # Check if new row
    if data_indices.x == 0:
        old_values_y = old_values_height
        old_values_height = data_indices.y + data_shape[H_DIM]
        old_values_x = 0

    # Check x-axis:
    # If this gradient is exactly on the border of old_value_indices
    # everything is new.
    if data_indices.x == old_values_x:
        rel_left = 0
        rel_right = data_shape[W_DIM]

    # If data_indices has some overlap with old_value_indices, trim unique
    # indices.
    else:
        assert old_values_x - data_indices.x >= 0, "Misses data in x-axis!"
        rel_left = old_values_x - data_indices.x
        rel_right = data_shape[W_DIM]

    # Check y-axis:
    # Equal to column logic (see above)
    if data_indices.y == old_values_y:
        rel_top = 0
        rel_bottom = data_shape[H_DIM]
    else:
        assert old_values_y - data_indices.y >= 0, "We miss data in y-axis"
        rel_top = old_values_y - data_indices.y
        rel_bottom = data_shape[H_DIM]

    # Update old-value-indices
    old_values_x += rel_right - rel_left

    assert rel_top >= 0, f"We miss data in y-axis before: {data_indices}"
    assert rel_left >= 0, f"We miss data in x-axis before: {data_indices}"

    new_value_indices = Box(rel_top, rel_bottom - rel_top, rel_left, rel_right - rel_left, None)
    old_value_indices = Box(int(old_values_y), int(old_values_height), int(old_values_x), 0, None)

    return new_value_indices, old_value_indices