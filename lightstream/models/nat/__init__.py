"""Neighborhood Attention Transformer models and NCHW building blocks."""

from lightstream.models.nat.nchw import (
    NCHWNATLayer,
    PointwiseConvMlp,
    convert_nchw_nat_state_dict,
    convert_nhwc_nat_state_dict,
    copy_nhwc_nat_to_nchw,
    linear_to_pointwise_conv,
    pointwise_conv_to_linear,
)

__all__ = [
    "NCHWNATLayer",
    "PointwiseConvMlp",
    "convert_nchw_nat_state_dict",
    "convert_nhwc_nat_state_dict",
    "copy_nhwc_nat_to_nchw",
    "linear_to_pointwise_conv",
    "pointwise_conv_to_linear",
]
