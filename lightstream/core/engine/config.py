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


@dataclass(frozen=True)
class TensorSpec:
    """Serializable metadata for one tensor produced by the compiled tile graph."""

    shape: tuple[int, ...]
    dtype: str | None = None
    device: str | None = None


@dataclass(frozen=True)
class InputSpec:
    """Input tensor contract used to compile the streaming plan."""

    tensor: TensorSpec
    model_signature: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TilePlan:
    """Tile-level geometry shared by forward and backward replay."""

    tile_shape: tuple[int, int, int, int]
    tile_output_shape: tuple[int, ...] | None = None
    tile_output_shapes: tuple[tuple[int, ...], ...] = ()
    tile_output_lost: tuple[Any, ...] = ()
    tile_gradient_lost: Any = None
    output_stride_per_output: tuple[Any, ...] = ()
    output_stride: Any = None


@dataclass(frozen=True)
class LayerPlan:
    """Streamability metadata for a converted streaming layer."""

    name: str
    module_type: str
    streamable: bool
    output_stride: Any = None
    lost: Any = None
    grad_lost: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReducerNode:
    """Compiled metadata for a streaming reducer node."""

    name: str
    reducer_type: str
    output_index: int | None = None
    input_indices: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputLayout:
    """Flattened internal output layout and public structure metadata."""

    tensor_specs: tuple[TensorSpec, ...]
    output_structure: object
    public_indices: tuple[int, ...] = ()
    reducer_auxiliary_indices: tuple[int, ...] = ()


@dataclass
class StreamingPlanState:
    """Mutable statistics captured while compilation is in progress."""

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


@dataclass(frozen=True)
class CompiledPlan:
    """Immutable compiled streaming execution plan plus mutable per-session replay state."""

    input_spec: InputSpec | None = None
    tile_plan: TilePlan | None = None
    output_layout: OutputLayout | None = None
    layer_plans: tuple[LayerPlan, ...] = ()
    reducer_nodes: tuple[ReducerNode, ...] = ()
    public_output_spec: object = None
    stream_network: Any = None
    session_state: StreamingSessionState = field(default_factory=StreamingSessionState, compare=False)
    plan_state: StreamingPlanState = field(default_factory=StreamingPlanState, compare=False)


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
    compiled_plan: CompiledPlan | None = None


@dataclass(frozen=True)
class BackwardContext:
    image: torch.Tensor
    grad_tensors: list
    tile_height: int
    tile_width: int
    output_heights: list
    output_widths: list
    compiled_plan: CompiledPlan | None = None
