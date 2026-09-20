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

from lightstream.core.layers import (
    ChannelLayerNorm,
    NeighborhoodAttention2D,
    StreamingMerge,
)


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


class NCHWConvTokenizer(nn.Module):
    """NCHW-native equivalent of NAT's convolutional patch tokenizer.

    Unlike the reference tokenizer, the public layout remains NCHW through
    both convolutions and the channel normalization.  Keeping ``proj`` and
    ``norm`` as the attribute names also preserves the reference checkpoint
    key layout.
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 96, *, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.norm = ChannelLayerNorm(embed_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


def copy_nhwc_conv_tokenizer_to_nchw(
    reference: nn.Module, target: NCHWConvTokenizer
) -> NCHWConvTokenizer:
    """Copy a reference :class:`ConvTokenizer` into its NCHW equivalent.

    Values as well as parameter dtype, device, and ``requires_grad`` state are
    retained.  A normalized reference tokenizer is required because the NCHW
    tokenizer always includes its final channel normalization.
    """

    if len(reference.proj) != 2 or not all(
        isinstance(layer, nn.Conv2d) for layer in reference.proj
    ):
        raise TypeError("reference.proj must contain exactly two Conv2d layers")
    if not isinstance(reference.norm, nn.LayerNorm):
        raise TypeError("reference.norm must be an nn.LayerNorm")
    if tuple(reference.norm.normalized_shape) != (target.norm.num_channels,):
        raise ValueError("reference and target tokenizer dimensions differ")

    source_parameters = list(reference.parameters())
    target_parameters = list(target.parameters())
    if len(source_parameters) != len(target_parameters):
        raise ValueError("reference and target tokenizer parameters differ")

    # Tokenizers normally have one dtype/device.  Assigning each copied tensor
    # separately additionally preserves deliberately mixed parameter setups.
    with torch.no_grad():
        for source, destination in zip(source_parameters, target_parameters):
            if source.shape != destination.shape:
                raise ValueError("reference and target tokenizer dimensions differ")
            destination.data = source.detach().clone()
            destination.requires_grad_(source.requires_grad)
    target.norm.eps = reference.norm.eps
    target.norm.norm.eps = reference.norm.eps
    target.train(reference.training)
    return target


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
                raise TypeError(
                    "`num_heads` is required when `attention` is not supplied"
                )
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
                raise ValueError(
                    "`num_heads` cannot be combined with an existing attention module"
                )
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


class ConvDownsampler(nn.Module):
    """NCHW equivalent of NAT's optional stage downsampler."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = ChannelLayerNorm(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.reduction(x))


class NCHWNATBlock(nn.Module):
    """A production NCHW NAT stage composed of :class:`NCHWNATLayer` objects.

    ``blocks`` and ``downsample`` match the reference NHWC implementation's
    names, keeping checkpoint keys stable apart from converted MLP weights.
    """

    def __init__(
        self,
        channels: int,
        depth: int,
        num_heads: int,
        kernel_size: int = 7,
        dilations: list[int] | tuple[int, ...] | None = None,
        *,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if depth < 0:
            raise ValueError("`depth` must be non-negative")
        if dilations is not None and len(dilations) != depth:
            raise ValueError("`dilations` must contain exactly `depth` values")

        self.channels = channels
        self.depth = depth
        self.blocks = nn.ModuleList(
            NCHWNATLayer(
                channels=channels,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=None if dilations is None else dilations[index],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
            )
            for index in range(depth)
        )
        self.downsample = ConvDownsampler(channels) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class NCHWNAT(nn.Module):
    """NCHW feature extractor corresponding to the four-stage NAT backbone.

    Global pooling and the classification head are intentionally left to the
    caller so the spatial feature map can be consumed by Lightstream.  NAT's
    stochastic regularizers are not tile invariant during streamed training,
    and are consequently rejected rather than silently producing different
    full-frame and streamed results.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float,
        depths: list[int] | tuple[int, ...],
        num_heads: list[int] | tuple[int, ...],
        drop_path_rate: float = 0.2,
        in_chans: int = 3,
        kernel_size: int = 7,
        dilations: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        stochastic = {
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "drop_path_rate": drop_path_rate,
        }
        unsupported = [name for name, value in stochastic.items() if value != 0]
        if unsupported:
            settings = ", ".join(f"{name}={stochastic[name]!r}" for name in unsupported)
            raise ValueError(
                "NCHWNAT requires drop_rate=0, attn_drop_rate=0, and "
                "drop_path_rate=0 for deterministic streamed training; "
                f"unsupported setting(s): {settings}"
            )
        if len(depths) != 4:
            raise ValueError("NCHWNAT requires exactly four stages")
        if len(num_heads) != len(depths):
            raise ValueError("`num_heads` must contain one value per stage")
        if dilations is not None and len(dilations) != len(depths):
            raise ValueError("`dilations` must contain one list per stage")

        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio

        # These names deliberately match NAT so the stage portions of existing
        # checkpoint keys do not need to be remapped.
        self.patch_embed = NCHWConvTokenizer(in_chans=in_chans, embed_dim=embed_dim)
        self.levels = nn.ModuleList(
            NCHWNATBlock(
                channels=int(embed_dim * 2**index),
                depth=depth,
                num_heads=num_heads[index],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[index],
                downsample=index < 3,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for index, depth in enumerate(depths)
        )
        self.norm = ChannelLayerNorm(self.num_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final normalized NCHW map without pooling or a head."""

        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def copy_nhwc_nat_to_nchw(reference: nn.Module, target: NCHWNATLayer) -> NCHWNATLayer:
    """Copy an original-layout NAT layer into an NCHW production layer."""

    target.norm1.norm.load_state_dict(reference.norm1.state_dict())
    target.attn.attention.load_state_dict(reference.attn.state_dict())
    target.norm2.norm.load_state_dict(reference.norm2.state_dict())
    target.mlp.load_state_dict(reference.mlp.state_dict())
    return target


def copy_nhwc_nat_block_to_nchw(
    reference: nn.Module, target: NCHWNATBlock
) -> NCHWNATBlock:
    """Copy every parameter of an original-layout NAT block to NCHW."""

    if len(reference.blocks) != len(target.blocks):
        raise ValueError("reference and target NAT blocks have different depths")
    for reference_layer, target_layer in zip(reference.blocks, target.blocks):
        copy_nhwc_nat_to_nchw(reference_layer, target_layer)

    if (reference.downsample is None) != (target.downsample is None):
        raise ValueError("reference and target must use the same downsample setting")
    if reference.downsample is not None:
        target.downsample.reduction.load_state_dict(
            reference.downsample.reduction.state_dict()
        )
        target.downsample.norm.norm.load_state_dict(
            reference.downsample.norm.state_dict()
        )
    return target


__all__ = [
    "ConvDownsampler",
    "NCHWConvDownsampler",
    "NCHWConvTokenizer",
    "NCHWNAT",
    "NCHWNATBlock",
    "NCHWNATLayer",
    "PointwiseConvMlp",
    "convert_nchw_nat_state_dict",
    "convert_nhwc_nat_state_dict",
    "copy_nhwc_nat_block_to_nchw",
    "copy_nhwc_nat_to_nchw",
    "copy_nhwc_conv_tokenizer_to_nchw",
    "linear_to_pointwise_conv",
    "pointwise_conv_to_linear",
]


# Kept as a compatibility alias for callers of the initial NCHW NAT API.  The
# unprefixed name is unambiguous inside this NCHW-only module and matches the
# corresponding NHWC production component.
NCHWConvDownsampler = ConvDownsampler
