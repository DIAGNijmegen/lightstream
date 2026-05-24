import warnings

warnings.warn(
    "lightstream.modules.reducer is deprecated; use lightstream.core.reducer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from lightstream.core.reducer import (  # noqa: E402
    MeanReducer,
    SumReducer,
    StreamingGeMReducer,
    StreamingMeanReducer,
    StreamingReducer,
    StreamingSumReducer,
)

__all__ = ["MeanReducer", "SumReducer", "StreamingReducer", "StreamingMeanReducer", "StreamingSumReducer", "StreamingGeMReducer"]
