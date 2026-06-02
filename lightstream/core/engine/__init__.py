"""Streaming engine package."""
from .config import (
    BackwardContext,
    CompiledPlan,
    ForwardContext,
    InputSpec,
    LayerPlan,
    OutputLayout,
    ReducerNode,
    StreamingConfig,
    TensorSpec,
    TilePlan,
    TileSpec,
)
from .scnn import StreamingCNN

__all__ = [
    "BackwardContext",
    "CompiledPlan",
    "ForwardContext",
    "InputSpec",
    "LayerPlan",
    "OutputLayout",
    "ReducerNode",
    "StreamingCNN",
    "StreamingConfig",
    "TensorSpec",
    "TilePlan",
    "StreamingEngine",
    "TileSpec",
]


def __getattr__(name):
    if name == "StreamingEngine":
        from .api import StreamingEngine

        return StreamingEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
