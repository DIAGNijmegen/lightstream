"""Public reducer API.

This plural package is the preferred import location for reducer extension
points. Concrete reducer implementations are loaded lazily to keep compatibility
with the legacy :mod:`lightstream.core.reducer` package during import.
"""

from .api import (
    BaseReducer,
    ManualBackwardReducer,
    ManualVJPReducer,
    MultiInputSpatialReducer,
    SpatialReducer,
    validate_aligned_nchw_inputs,
    validate_arity,
    validate_channel_compatibility,
    validate_mask_shape,
    validate_nchw_shape,
)
from .base import (
    BaseStreamingGlobalReducer,
    ReducerMeta,
    ReducerReplayRecord,
    ReducerTile,
    StreamingReducer,
    streaming_reduce_tile,
)

_LAZY_REDUCER_EXPORTS = {
    "MeanReducer",
    "SumReducer",
    "GeMReducer",
    "AttentionGeMReducer",
    "StreamingMeanReducer",
    "StreamingSumReducer",
    "StreamingGeMReducer",
    "StreamingAttentionGeMReducer",
}


def __getattr__(name: str):
    if name in _LAZY_REDUCER_EXPORTS:
        from lightstream.core import reducer as _legacy_reducer

        value = getattr(_legacy_reducer, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MeanReducer",
    "SumReducer",
    "GeMReducer",
    "AttentionGeMReducer",
    "BaseReducer",
    "SpatialReducer",
    "MultiInputSpatialReducer",
    "ManualVJPReducer",
    "ManualBackwardReducer",
    "BaseStreamingGlobalReducer",
    "StreamingReducer",
    "ReducerMeta",
    "ReducerTile",
    "ReducerReplayRecord",
    "streaming_reduce_tile",
    "StreamingMeanReducer",
    "StreamingSumReducer",
    "StreamingGeMReducer",
    "StreamingAttentionGeMReducer",
    "validate_arity",
    "validate_nchw_shape",
    "validate_channel_compatibility",
    "validate_aligned_nchw_inputs",
    "validate_mask_shape",
]
