"""Streaming engine package."""
from .config import BackwardContext, CompiledPlan, ForwardContext, StreamingConfig, TileSpec
from .scnn import StreamingCNN

__all__ = [
    "BackwardContext",
    "CompiledPlan",
    "ForwardContext",
    "StreamingCNN",
    "StreamingConfig",
    "StreamingEngine",
    "TileSpec",
]


def __getattr__(name):
    if name == "StreamingEngine":
        from .api import StreamingEngine

        return StreamingEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
