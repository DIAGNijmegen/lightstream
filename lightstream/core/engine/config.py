"""Shared runtime configuration and tile context types for the streaming engine."""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeAlias

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
    mean: Optional[tuple[float, float, float]] = None
    std: Optional[tuple[float, float, float]] = None
    add_keep_modules: Optional[list[type[torch.nn.Module]]] = None
    before_streaming_init_callbacks: Optional[list[Callable[..., Any]]] = None
    after_streaming_init_callbacks: Optional[list[Callable[..., Any]]] = None


@dataclass
class StreamingPlanState:
    """Mutable statistics captured by compilation and reused by tile execution."""

    tile_output_shape: Any = None
    tile_output_shapes: Any = None
    tile_output_lost: Any = None
    output_stride_per_output: Any = None
    output_spec: Any = None


@dataclass
class StreamingSessionState:
    """Mutable state that belongs to a single forward/backward streaming session."""

    reducer_head_map: dict[int, Any] = field(default_factory=dict)
    reducer_input_indices: dict[int, Any] = field(default_factory=dict)
    last_forward_tiles: list[Any] = field(default_factory=list)
    active_reducer_mask: Any = None


@dataclass
class CompiledPlan:
    """Compiled streaming execution plan and per-session mutable state."""

    stream_network: Any = None
    plan_state: StreamingPlanState = field(default_factory=StreamingPlanState)
    session_state: StreamingSessionState = field(default_factory=StreamingSessionState)


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
