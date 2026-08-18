"""Compatibility aliases for geometry helpers formerly defined here."""

from typing import NamedTuple

from lightstream.core.engine.geometry import (
    B_DIM,
    C_DIM,
    H_DIM,
    W_DIM,
    Box,
    Lost,
    Sides,
    _new_value_indices,
    _ntuple,
)

_triple = _ntuple(3)


class IOShape(NamedTuple):
    batch: int
    channels: int
    height: int
    width: int


__all__ = [
    "B_DIM", "C_DIM", "H_DIM", "W_DIM", "Box", "IOShape", "Lost", "Sides",
    "_new_value_indices", "_ntuple", "_triple",
]
