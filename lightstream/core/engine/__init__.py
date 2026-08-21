"""Planning and execution primitives for tiled neural-network streaming."""

from .configuration import HeadPlan, ModulePlan, StreamingPlan, TilePlan
from .cache import CACHE_VERSION, PlanCacheError, load_plan, migrate_legacy_cache, serialize_plan
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
    "CACHE_VERSION",
    "Box",
    "ForwardExecutor",
    "HeadPlan",
    "Lost",
    "ModulePlan",
    "PlanCacheError",
    "Sides",
    "StreamingPlan",
    "StreamingPlanBuilder",
    "StreamSession",
    "TilePlan",
    "load_plan",
    "migrate_legacy_cache",
    "serialize_plan",
]
