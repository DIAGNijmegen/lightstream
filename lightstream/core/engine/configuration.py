"""Immutable configuration produced before streaming execution."""

from dataclasses import dataclass
from typing import Any

from .geometry import Lost
from .reducers import StaticReducerBinding

OutputSpec = tuple[str, Any]


@dataclass(frozen=True)
class HeadPlan:
    tile_output_shape: tuple[int, ...]
    stride: tuple[int, ...]
    loss: Lost


@dataclass(frozen=True)
class ModulePlan:
    name: str
    module_type: str
    statistics: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class TilePlan:
    input_shape: tuple[int, ...]
    gradient_loss: Lost
    internal_alignment: tuple[int, int]


@dataclass(frozen=True)
class StreamingPlan:
    """Stable setup results shared by forward and backward executors."""

    tile: TilePlan
    heads: tuple[HeadPlan, ...]
    modules: tuple[ModulePlan, ...]
    reducer_heads: tuple[StaticReducerBinding, ...]
    output_structure: OutputSpec
