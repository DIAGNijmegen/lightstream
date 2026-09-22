"""NCHW-native Neighborhood Attention Transformer building blocks and models.

The implementation keeps feature maps in NCHW layout, uses pointwise
convolutions for MLP projections, and supports deterministic tiled execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number

import torch
from torch import nn

from lightstream.core.layers import (
    ChannelLayerNorm,
    LayerScale,
    NeighborhoodAttention2D,
    StreamingMerge,
)

nchw_model_urls = {
    "nat_mini_1k": "https://huggingface.co/itsjustafleshwound/nat-mini-nchw/resolve/main/nat_mini-nchw.pth",
    "nat_tiny_1k": "https://huggingface.co/itsjustafleshwound/nat-tiny-nchw/resolve/main/nat_tiny-nchw.pth",
    "nat_small_1k": "https://huggingface.co/itsjustafleshwound/nat-small-nchw/resolve/main/nat_small-nchw.pth",
    "nat_base_1k": "https://huggingface.co/itsjustafleshwound/nat-base-nchw/resolve/main/nat_base-nchw.pth",
}


def _load_pretrained_nchw(
    model: "NCHWNAT",
    pretrained: bool,
    checkpoint: str | None,
) -> "NCHWNAT":
    """Download and strictly load a hosted, original-keyed NCHW checkpoint."""

    if pretrained is False or pretrained is None:
        return model

    if pretrained is not True:
        raise TypeError("`pretrained` must be a bool")
    if checkpoint is None:
        raise ValueError("no pretrained checkpoint exists for this NAT variant")

    downloaded = torch.hub.load_state_dict_from_url(
        nchw_model_urls[checkpoint], map_location="cpu"
    )
    if not isinstance(downloaded, Mapping):
        raise ValueError("malformed NCHW checkpoint: expected a state-dict mapping")
    containers = [
        downloaded[key]
        for key in ("backbone_state_dict", "state_dict")
        if key in downloaded and isinstance(downloaded[key], Mapping)
    ]
    state_dict = containers[0] if containers else downloaded
    if not state_dict or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("malformed NCHW checkpoint: no tensor state dictionary found")
    state_dict = dict(state_dict)
    state_dict.pop("head.weight", None)
    state_dict.pop("head.bias", None)
    model.load_state_dict(state_dict, strict=True)
    return model


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
        layer_scale: float | None = None,
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
        if layer_scale is not None and not isinstance(layer_scale, Number):
            raise TypeError("`layer_scale` must be numeric or None")
        if layer_scale is not None:
            self.gamma1 = LayerScale((1, channels, 1, 1), layer_scale)
        self.merge1 = StreamingMerge("add")
        self.norm2 = ChannelLayerNorm(channels)
        self.mlp = PointwiseConvMlp(channels, hidden_channels)
        if layer_scale is not None:
            self.gamma2 = LayerScale((1, channels, 1, 1), layer_scale)
        self.merge2 = StreamingMerge("add")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.attn(self.norm1(x))
        if hasattr(self, "gamma1"):
            attention = self.gamma1(attention)
        x = self.merge1(x, attention)
        mlp = self.mlp(self.norm2(x))
        if hasattr(self, "gamma2"):
            mlp = self.gamma2(mlp)
        return self.merge2(x, mlp)


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
        layer_scale: float | None = None,
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
                layer_scale=layer_scale,
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
    stochastic regularizers are not tile invariant during streamed training.
    They are intentionally unsupported for deterministic tiled training and
    must remain zero, so nonzero values are rejected rather than silently
    producing different full-frame and streamed results.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float,
        depths: list[int] | tuple[int, ...],
        num_heads: list[int] | tuple[int, ...],
        drop_path_rate: float = 0.0,
        in_chans: int = 3,
        kernel_size: int = 7,
        dilations: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        layer_scale: float | None = None,
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
        # Original NAT uses nn.LayerNorm's 1e-5 default throughout.
        norm_eps = 1e-5
        self.patch_embed = NCHWConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, eps=norm_eps
        )
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
                layer_scale=layer_scale,
            )
            for index, depth in enumerate(depths)
        )
        self.norm = ChannelLayerNorm(self.num_features)

        # Nested building blocks retain their historical 1e-6 default when
        # used independently; a complete NCHWNAT must mirror original NAT.
        for module in self.modules():
            if isinstance(module, ChannelLayerNorm):
                module.eps = norm_eps
                module.norm.eps = norm_eps

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final normalized NCHW map without pooling or a head."""

        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def nchw_nat_mini(pretrained: bool = False, **kwargs) -> NCHWNAT:
    """Build the deterministic NCHW counterpart of :func:`nat_mini`.

    The original factory's stochastic-depth default is intentionally replaced
    with zero: stochastic regularizers are not invariant to Lightstream's tile
    replay.  Keeping the complete production configuration in one public
    factory also prevents checkpoint conversion tests and applications from
    silently drifting apart.
    """

    model = NCHWNAT(
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        kernel_size=7,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale=None,
        **kwargs,
    )
    return _load_pretrained_nchw(model, pretrained, "nat_mini_1k")


def nchw_nat_tiny(pretrained: bool = False, **kwargs) -> NCHWNAT:
    """Build the deterministic NCHW counterpart of :func:`nat_tiny`."""

    model = NCHWNAT(
        depths=[3, 4, 18, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        kernel_size=7,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale=None,
        **kwargs,
    )
    return _load_pretrained_nchw(model, pretrained, "nat_tiny_1k")


def nchw_nat_nano(pretrained: bool = False, **kwargs) -> NCHWNAT:
    """Build the deterministic NCHW synthetic Nano NAT variant."""

    model = NCHWNAT(
        depths=[3, 4, 6, 5],
        num_heads=[1, 2, 4, 8],
        embed_dim=32,
        mlp_ratio=2,
        kernel_size=7,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale=None,
        **kwargs,
    )
    return _load_pretrained_nchw(model, pretrained, None)


def nchw_nat_pico(pretrained: bool = False, **kwargs) -> NCHWNAT:
    """Build the deterministic NCHW synthetic Pico NAT variant."""

    model = NCHWNAT(
        depths=[3, 4, 6, 5],
        num_heads=[1, 2, 4, 8],
        embed_dim=16,
        mlp_ratio=2,
        kernel_size=7,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale=None,
        **kwargs,
    )
    return _load_pretrained_nchw(model, pretrained, None)


def nchw_nat_small(pretrained: bool = False, **kwargs) -> NCHWNAT:
    """Build the deterministic NCHW counterpart of :func:`nat_small`."""

    model = NCHWNAT(
        depths=[3, 4, 18, 5],
        num_heads=[3, 6, 12, 24],
        embed_dim=96,
        mlp_ratio=2,
        kernel_size=7,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale=1e-5,
        **kwargs,
    )
    return _load_pretrained_nchw(model, pretrained, "nat_small_1k")


def nchw_nat_base(pretrained: bool = False, **kwargs) -> NCHWNAT:
    """Build the deterministic NCHW counterpart of :func:`nat_base`."""

    model = NCHWNAT(
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=2,
        kernel_size=7,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale=1e-5,
        **kwargs,
    )
    return _load_pretrained_nchw(model, pretrained, "nat_base_1k")


# Backward-compatible names from the original NCHW public API.
NCHWNatMini = nchw_nat_mini
NCHWNatNano = nchw_nat_nano
NCHWNatPico = nchw_nat_pico
NCHWNatSmall = nchw_nat_small
NCHWNatBase = nchw_nat_base

__all__ = [
    "ConvDownsampler",
    "NCHWConvDownsampler",
    "NCHWConvTokenizer",
    "NCHWNAT",
    "NCHWNatMini",
    "NCHWNatNano",
    "NCHWNatPico",
    "NCHWNatSmall",
    "NCHWNatBase",
    "nchw_nat_mini",
    "nchw_nat_tiny",
    "nchw_nat_small",
    "nchw_nat_base",
    "nchw_nat_nano",
    "nchw_nat_pico",
    "NCHWNATBlock",
    "NCHWNATLayer",
    "PointwiseConvMlp",
]

# Kept as a compatibility alias for callers of the initial NCHW NAT API.  The
# unprefixed name is unambiguous inside this NCHW-only module and matches the
# corresponding NHWC production component.
NCHWConvDownsampler = ConvDownsampler
