import warnings

warnings.warn(
    "lightstream.modules.reducer is deprecated; use lightstream.core.reducer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from lightstream.core.reducer import (  # noqa: E402
    BaseReducer,
    GeMReducer,
    MeanReducer,
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
    "StreamingReducer",
    "StreamingMeanReducer",
    "StreamingSumReducer",
    "StreamingGeMReducer",
]
