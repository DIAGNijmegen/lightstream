"""Public reducer API.

This plural package is the preferred import location for reducer extension
points. The legacy :mod:`lightstream.core.reducer` package re-exports the same
symbols for compatibility.
"""

from lightstream.core.reducer import (
    AttentionGeMReducer,
    BaseReducer,
    BaseStreamingGlobalReducer,
    GeMReducer,
    ManualBackwardReducer,
    ManualVJPReducer,
    MeanReducer,
    MultiInputSpatialReducer,
    ReducerMeta,
    ReducerReplayRecord,
    ReducerTile,
    SpatialReducer as OfflineSpatialReducer,
    StreamingAttentionGeMReducer,
    StreamingGeMReducer,
    StreamingMeanReducer,
    StreamingReducer,
    StreamingSumReducer,
    SumReducer,
)
from .base import SpatialReducer

__all__ = [
    "MeanReducer",
    "SumReducer",
    "GeMReducer",
    "AttentionGeMReducer",
    "BaseReducer",
    "SpatialReducer",
    "OfflineSpatialReducer",
    "MultiInputSpatialReducer",
    "ManualVJPReducer",
    "ManualBackwardReducer",
    "BaseStreamingGlobalReducer",
    "StreamingReducer",
    "ReducerMeta",
    "ReducerTile",
    "ReducerReplayRecord",
    "StreamingMeanReducer",
    "StreamingSumReducer",
    "StreamingGeMReducer",
    "StreamingAttentionGeMReducer",
]
