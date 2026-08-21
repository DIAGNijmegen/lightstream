"""Planning and execution primitives for tiled neural-network streaming."""

from .configuration import HeadPlan, ModulePlan, StreamingPlan, TilePlan
from .executors import BackwardExecutor, ForwardExecutor
from .geometry import Box, Lost, Sides
from .planner import StreamingPlanBuilder, UnsupportedStreamingOperatorError
from .operators import (
    OperatorCapabilities,
    STREAMING_OPERATORS,
    StreamingOperatorAdapter,
    StreamingOperatorRegistry,
    register_operator,
)
from .session import StreamSession

__all__ = [
    "BackwardExecutor",
    "Box",
    "ForwardExecutor",
    "HeadPlan",
    "Lost",
    "ModulePlan",
    "Sides",
    "StreamingPlan",
    "StreamingPlanBuilder",
    "StreamSession",
    "TilePlan",
]
