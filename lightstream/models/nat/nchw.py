"""NCHW building blocks for Neighborhood Attention Transformer layers.

The conversion helpers in this module provide an explicit checkpoint boundary
between the original NHWC NAT representation (``nn.Linear`` MLP projections)
and Lightstream's NCHW representation (pointwise ``nn.Conv2d`` projections).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn

from lightstream.core.layers import ChannelLayerNorm, NeighborhoodAttention2D, StreamingMerge


def linear_to_pointwise_conv(linear: nn.Linear) -> nn.Conv2d:
    """Return a 1x1 convolution equivalent to ``linear``.

    Parameter values, dtype, device, and per-parameter ``requires_grad`` flags
    are retained.  The returned module does not share parameters with the
    input module.
    """

    if not isinstance(linear, nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(linear).__name__}")
    convolution = nn.Conv2d(
        linear.in_features,
        linear.out_features,
        kernel_size=1,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        convolution.weight.copy_(linear.weight[:, :, None, None])
        if linear.bias is not None:
            convolution.bias.copy_(linear.bias)
    convolution.weight.requires_grad_(linear.weight.requires_grad)
    if linear.bias is not None:
        convolution.bias.requires_grad_(linear.bias.requires_grad)
    convolution.train(linear.training)
    return convolution


def pointwise_conv_to_linear(convolution: nn.Conv2d) -> nn.Linear:
    """Return an ``nn.Linear`` equivalent to a 1x1 convolution."""

    if not isinstance(convolution, nn.Conv2d) or convolution.kernel_size != (1, 1):
        raise TypeError("expected an nn.Conv2d with kernel_size=1")
    if convolution.groups != 1:
        raise ValueError("a grouped convolution cannot be represented by nn.Linear")
    linear = nn.Linear(
        convolution.in_channels,
        convolution.out_channels,
        bias=convolution.bias is not None,
        device=convolution.weight.device,
        dtype=convolution.weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(convolution.weight[:, :, 0, 0])
        if convolution.bias is not None:
            linear.bias.copy_(convolution.bias)
    linear.weight.requires_grad_(convolution.weight.requires_grad)
    if convolution.bias is not None:
        linear.bias.requires_grad_(convolution.bias.requires_grad)
    linear.train(convolution.training)
    return linear


def convert_nhwc_nat_state_dict(state_dict: Mapping[str, torch.Tensor]):
    """Convert NHWC NAT Linear weights into NCHW pointwise-convolution weights.

    Keys are unchanged; only two-dimensional tensors whose key ends in
    ``mlp.fc1.weight`` or ``mlp.fc2.weight`` gain two singleton dimensions.
    This makes original NAT checkpoints loadable without manual reshaping.
    """

    converted = OrderedDict()
    for key, value in state_dict.items():
        if (
            key.endswith(("mlp.fc1.weight", "mlp.fc2.weight"))
            and isinstance(value, torch.Tensor)
            and value.ndim == 2
        ):
            value = value[:, :, None, None]
        converted[key] = value
    if hasattr(state_dict, "_metadata"):
        converted._metadata = state_dict._metadata
    return converted


def convert_nchw_nat_state_dict(state_dict: Mapping[str, torch.Tensor]):
    """Convert NCHW pointwise MLP weights back to original NHWC NAT shape."""

    converted = OrderedDict()
    for key, value in state_dict.items():
        if (
            key.endswith(("mlp.fc1.weight", "mlp.fc2.weight"))
            and isinstance(value, torch.Tensor)
            and value.ndim == 4
        ):
            if value.shape[-2:] != (1, 1):
                raise ValueError(f"{key!r} is not a pointwise-convolution weight")
            value = value[:, :, 0, 0]
        converted[key] = value
    if hasattr(state_dict, "_metadata"):
        converted._metadata = state_dict._metadata
    return converted


class PointwiseConvMlp(nn.Module):
    """NCHW equivalent of NAT's two-linear-layer pointwise MLP."""

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Keep the original NAT key names stable and accept its Linear tensor
        # shapes directly. PyTorch copies these temporary views into parameters.
        for projection in ("fc1", "fc2"):
            key = f"{prefix}{projection}.weight"
            value = state_dict.get(key)
            if value is not None and value.ndim == 2:
                state_dict[key] = value[:, :, None, None]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class NCHWNATLayer(nn.Module):
    """A NAT layer with an NCHW public layout and explicit residual merges.

    An existing NATTEN attention module may be supplied for model conversion.
    Otherwise the wrapped production operator is constructed lazily from the
    explicit NATTEN configuration arguments.
    """

    def __init__(
        self,
        attention: nn.Module | None = None,
        channels: int | None = None,
        hidden_channels: int | None = None,
        *,
        num_heads: int | None = None,
        kernel_size: int = 7,
        dilation: int | None = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if channels is None:
            raise TypeError("`channels` is required")
        if hidden_channels is None:
            hidden_channels = int(channels * mlp_ratio)
        if attention is None:
            if num_heads is None:
                raise TypeError("`num_heads` is required when `attention` is not supplied")
            wrapped_attention = NeighborhoodAttention2D(
                dim=channels,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                rel_pos_bias=True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            if num_heads is not None:
                raise ValueError("`num_heads` cannot be combined with an existing attention module")
            wrapped_attention = NeighborhoodAttention2D(attention=attention)

        self.norm1 = ChannelLayerNorm(channels)
        self.attn = wrapped_attention
        self.merge1 = StreamingMerge("add")
        self.norm2 = ChannelLayerNorm(channels)
        self.mlp = PointwiseConvMlp(channels, hidden_channels)
        self.merge2 = StreamingMerge("add")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.merge1(x, self.attn(self.norm1(x)))
        return self.merge2(x, self.mlp(self.norm2(x)))


def copy_nhwc_nat_to_nchw(reference: nn.Module, target: NCHWNATLayer) -> NCHWNATLayer:
    """Copy an original-layout NAT layer into an NCHW production layer."""

    target.norm1.norm.load_state_dict(reference.norm1.state_dict())
    target.attn.attention.load_state_dict(reference.attn.state_dict())
    target.norm2.norm.load_state_dict(reference.norm2.state_dict())
    target.mlp.load_state_dict(reference.mlp.state_dict())
    return target


__all__ = [
    "NCHWNATLayer",
    "PointwiseConvMlp",
    "convert_nchw_nat_state_dict",
    "convert_nhwc_nat_state_dict",
    "copy_nhwc_nat_to_nchw",
    "linear_to_pointwise_conv",
    "pointwise_conv_to_linear",
]
