"""Streaming engine package."""
from .adapters import AdapterRegistry, StreamableLayerAdapter
from .config import BackwardContext, CompiledPlan, ForwardContext, StreamingConfig, TileSpec
from .orchestration import BackwardExecutor, ForwardExecutor, ReducerRuntime, TilePlanner
from .scnn import StreamingCNN

__all__ = [
    "AdapterRegistry",
    "BackwardContext",
    "BackwardExecutor",
    "CompiledPlan",
    "ForwardContext",
    "ForwardExecutor",
    "StreamingCNN",
    "ReducerRuntime",
    "StreamableLayerAdapter",
    "StreamingConfig",
    "TensorSpec",
    "TilePlan",
    "StreamingEngine",
    "TilePlanner",
    "TileSpec",
]


def __getattr__(name):
    if name == "StreamingEngine":
        from .api import StreamingEngine

        return StreamingEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
