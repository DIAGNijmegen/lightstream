"""Public streamed Neighborhood Attention Transformer feature extractors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from lightstream.core.layers import NeighborhoodAttention2D
from lightstream.models.nat.nat import NAT, model_urls
from lightstream.models.nat.nchw import (
    NCHWNatBase,
    NCHWNatMini,
    NCHWNatSmall,
    convert_nhwc_nat_state_dict,
)
from lightstream.modules.streaming import StreamingModule


_VARIANTS = {
    "nat_mini": {
        "reference": dict(
            depths=[3, 4, 6, 5], num_heads=[2, 4, 8, 16],
            embed_dim=64, mlp_ratio=3, kernel_size=7, layer_scale=None,
        ),
        "nchw": NCHWNatMini,
        "checkpoint": "nat_mini_1k",
    },
    "nat_small": {
        "reference": dict(
            depths=[3, 4, 18, 5], num_heads=[3, 6, 12, 24],
            embed_dim=96, mlp_ratio=2, kernel_size=7, layer_scale=1e-5,
        ),
        "nchw": NCHWNatSmall,
        "checkpoint": "nat_small_1k",
    },
    "nat_base": {
        "reference": dict(
            depths=[3, 4, 18, 5], num_heads=[4, 8, 16, 32],
            embed_dim=128, mlp_ratio=2, kernel_size=7, layer_scale=1e-5,
        ),
        "nchw": NCHWNatBase,
        "checkpoint": "nat_base_1k",
    },
}


def _checkpoint_state(
    variant: str,
    pretrained: bool | str | Path | Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor] | None:
    """Resolve the public checkpoint selection into an ordinary state dict."""

    if pretrained is False or pretrained is None:
        return None
    if isinstance(pretrained, Mapping):
        state = pretrained
    elif pretrained is True:
        state = torch.hub.load_state_dict_from_url(
            model_urls[_VARIANTS[variant]["checkpoint"]], map_location="cpu"
        )
    elif isinstance(pretrained, (str, Path)):
        selection = str(pretrained)
        if selection in model_urls:
            state = torch.hub.load_state_dict_from_url(
                model_urls[selection], map_location="cpu"
            )
        elif selection.startswith(("http://", "https://")):
            state = torch.hub.load_state_dict_from_url(selection, map_location="cpu")
        else:
            state = torch.load(selection, map_location="cpu", weights_only=True)
    else:
        raise TypeError("`pretrained` must be a bool, checkpoint name/path, or state dict")

    # Accommodate the common checkpoint container without weakening strict load.
    if "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    return state


class StreamingNAT(StreamingModule):
    """Stream a deterministic NAT backbone and return its normalized NCHW map.

    ``pretrained=True`` selects the official checkpoint for ``variant``.  A
    key from :data:`model_urls`, local checkpoint path, URL, or state dictionary
    may be supplied instead.  Classification and global pooling are deliberately
    omitted from this wrapper.
    """

    def __init__(
        self,
        variant: str,
        tile_size: int,
        *,
        pretrained: bool | str | Path | Mapping[str, torch.Tensor] = True,
        tile_cache_path: str | Path | None = None,
        tile_cache: dict[str, Any] | None = None,
        tile_cache_state: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
        verbose: bool = True,
        deterministic: bool = True,
        saliency: bool = False,
        diagnose_saliency_assembly: bool = False,
        copy_to_gpu: bool = False,
        statistics_on_cpu: bool = True,
        normalize_on_gpu: bool = True,
        mean: list[float] | tuple[float, ...] | None = None,
        std: list[float] | tuple[float, ...] | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        defer_prepare: bool = False,
    ):
        if variant not in _VARIANTS:
            raise ValueError(
                f"Invalid NAT variant {variant!r}. Choose one of: {', '.join(_VARIANTS)}"
            )
        stochastic = {
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "drop_path_rate": drop_path_rate,
        }
        invalid = [f"{name}={value!r}" for name, value in stochastic.items() if value != 0]
        if invalid:
            raise ValueError(
                "StreamingNAT requires drop_rate=0, attn_drop_rate=0, and "
                "drop_path_rate=0; unsupported setting(s): " + ", ".join(invalid)
            )
        if tile_cache is not None and tile_cache_state is not None:
            raise ValueError("pass only one of `tile_cache` and `tile_cache_state`")
        if tile_cache_state is not None:
            tile_cache = tile_cache_state
        if tile_cache is not None and tile_cache_path is not None:
            raise ValueError("tile-cache state and `tile_cache_path` are mutually exclusive")
        if tile_cache is None and tile_cache_path is None:
            tile_cache_path = Path.cwd() / (
                f"{variant}_tile_cache_1_3_{tile_size}_{tile_size}"
            )

        config = _VARIANTS[variant]
        reference = NAT(
            **config["reference"], num_classes=1000,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        )
        checkpoint = _checkpoint_state(variant, pretrained)
        if checkpoint is not None:
            reference.load_state_dict(checkpoint, strict=True)

        # The head is intentionally not part of the streamed feature extractor.
        feature_state = {
            key: value for key, value in reference.state_dict().items()
            if not key.startswith("head.")
        }
        network = config["nchw"]()
        network.load_state_dict(convert_nhwc_nat_state_dict(feature_state), strict=True)
        if device is not None:
            network.to(device)

        self._provided_tile_cache = tile_cache
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        super().__init__(
            network, tile_size, tile_cache_path, defer_prepare=defer_prepare,
            add_keep_modules=[NeighborhoodAttention2D],
            verbose=verbose, deterministic=deterministic, saliency=saliency,
            diagnose_saliency_assembly=diagnose_saliency_assembly,
            copy_to_gpu=copy_to_gpu, statistics_on_cpu=statistics_on_cpu,
            normalize_on_gpu=normalize_on_gpu, mean=mean, std=std,
        )

    @staticmethod
    def get_model_names() -> list[str]:
        """Return supported public variant names."""

        return list(_VARIANTS)

    def load_tile_cache_if_needed(self, use_tile_cache: bool = True):
        if self._provided_tile_cache is not None:
            return self._provided_tile_cache
        return super().load_tile_cache_if_needed(use_tile_cache=use_tile_cache)

    def save_tile_cache_if_needed(self, overwrite: bool = False):
        if self._provided_tile_cache is None:
            return super().save_tile_cache_if_needed(overwrite=overwrite)
        return None


__all__ = ["StreamingNAT"]
