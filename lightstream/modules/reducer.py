import warnings

warnings.warn(
    "lightstream.modules.reducer is deprecated; use lightstream.core.reducer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from lightstream.core.reducer import (  # noqa: E402
    BaseReducer,
    FusedAttentionGeMReducer,
    GeMReducer,
    MeanReducer,
    StreamingFusedAttentionGeMReducer,
    StreamingGeMReducer,
    StreamingMeanReducer,
    StreamingReducer,
    StreamingSumReducer,
    SumReducer,
)

__all__ = [
    "BaseReducer",
    "MeanReducer",
    "SumReducer",
    "GeMReducer",
    "FusedAttentionGeMReducer",
    "StreamingReducer",
    "StreamingMeanReducer",
    "StreamingSumReducer",
    "StreamingGeMReducer",
    "StreamingFusedAttentionGeMReducer",
]
