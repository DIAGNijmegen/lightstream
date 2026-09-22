"""Neighborhood Attention Transformer models and NCHW building blocks."""

from lightstream.models.nat.nchw import (
    ConvDownsampler,
    NCHWConvDownsampler,
    NCHWConvTokenizer,
    NCHWNAT,
    NCHWNatMini,
    NCHWNatSmall,
    NCHWNatBase,
    NCHWNATBlock,
    NCHWNATLayer,
    PointwiseConvMlp,
    convert_nchw_nat_state_dict,
    convert_nhwc_nat_state_dict,
    copy_nhwc_nat_block_to_nchw,
    copy_nhwc_nat_model_to_nchw,
    copy_nhwc_nat_to_nchw,
    copy_nhwc_conv_tokenizer_to_nchw,
    linear_to_pointwise_conv,
    pointwise_conv_to_linear,
)
from lightstream.models.nat.streaming import StreamingNAT

StreamingNat = StreamingNAT

__all__ = [
    "ConvDownsampler",
    "NCHWConvDownsampler",
    "NCHWConvTokenizer",
    "NCHWNAT",
    "NCHWNatMini",
    "NCHWNatSmall",
    "NCHWNatBase",
    "NCHWNATBlock",
    "NCHWNATLayer",
    "PointwiseConvMlp",
    "convert_nchw_nat_state_dict",
    "convert_nhwc_nat_state_dict",
    "copy_nhwc_nat_block_to_nchw",
    "copy_nhwc_nat_model_to_nchw",
    "copy_nhwc_nat_to_nchw",
    "copy_nhwc_conv_tokenizer_to_nchw",
    "linear_to_pointwise_conv",
    "pointwise_conv_to_linear",
    "StreamingNAT",
    "StreamingNat",
]
