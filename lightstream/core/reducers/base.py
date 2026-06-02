"""Preferred reducer extension API."""

from lightstream.core.reducer.base import (
    BaseStreamingGlobalReducer,
    ReducerMeta,
    ReducerReplayRecord,
    ReducerTile,
    StreamingReducer,
    streaming_reduce_tile,
)

# ``SpatialReducer`` is the preferred name for authors implementing the new
# streaming contract: init_state(meta), update(state, tile), finalize(state).
SpatialReducer = BaseStreamingGlobalReducer

__all__ = [
    "SpatialReducer",
    "BaseStreamingGlobalReducer",
    "StreamingReducer",
    "ReducerMeta",
    "ReducerTile",
    "ReducerReplayRecord",
    "streaming_reduce_tile",
]
