"""Compatibility re-exports for reducer extension base classes."""

from lightstream.core.reducers.api import (
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

__all__ = [
    "BaseReducer",
    "SpatialReducer",
    "MultiInputSpatialReducer",
    "ManualVJPReducer",
    "ManualBackwardReducer",
    "validate_arity",
    "validate_nchw_shape",
    "validate_channel_compatibility",
    "validate_aligned_nchw_inputs",
    "validate_mask_shape",
]
