"""Shared runtime configuration and tile context types for the streaming engine."""
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

OutputSpec: TypeAlias = tuple[str, Any]
OutputStructure: TypeAlias = torch.Tensor | tuple[Any, ...] | list[Any] | dict[Any, Any]


@dataclass(frozen=True)
class StreamingConfig:
    """Runtime options captured when constructing a streaming engine."""

    tile_shape: tuple[int, int, int, int]
    verbose: bool = False
    deterministic: bool = False
    saliency: bool = False
    eps: float = 1e-5
    copy_to_gpu: bool = True
    statistics_on_cpu: bool = False
    normalize_on_gpu: bool = False


@dataclass(frozen=True)
class TileSpec:
    """Input-space description for a streaming tile."""

    y: int
    x: int
    height: int
    width: int
    sides: Any


@dataclass(frozen=True)
class ForwardContext:
    image: torch.Tensor
    tile_height: int
    tile_width: int
    output_heights: list
    output_widths: list
    valid_input_height: int
    valid_input_width: int
    n_rows: int
    n_cols: int
    result_device: torch.device


@dataclass(frozen=True)
class BackwardContext:
    image: torch.Tensor
    grad_tensors: list
    tile_height: int
    tile_width: int
    output_heights: list
    output_widths: list
