"""Public streamed Neighborhood Attention Transformer feature extractors."""

from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from lightstream.core.layers import NeighborhoodAttention2D
from lightstream.models.nat.nchw import (
    nchw_nat_base,
    nchw_nat_mini,
    nchw_nat_nano,
    nchw_nat_pico,
    nchw_nat_small,
    nchw_nat_tiny,
)
from lightstream.modules.streaming import StreamingModule


class StreamingNAT(StreamingModule):
    """Stream a deterministic NCHW NAT backbone."""

    def __init__(
        self,
        encoder: str,
        tile_size: int,
        *,
        pretrained: bool = True,
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
        model_choices = self.get_model_choices()
        if encoder not in model_choices:
            raise ValueError(
                f"Invalid model name {encoder!r}. Choose one of: "
                + ", ".join(model_choices)
            )
        stochastic = {
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "drop_path_rate": drop_path_rate,
        }
        invalid = [f"{key}={value!r}" for key, value in stochastic.items() if value]
        if invalid:
            raise ValueError(
                "StreamingNAT requires all stochastic rates to be zero; "
                + ", ".join(invalid)
            )
        if tile_cache is not None and tile_cache_state is not None:
            raise ValueError("pass only one of `tile_cache` and `tile_cache_state`")
        if tile_cache_state is not None:
            tile_cache = tile_cache_state
        if tile_cache is not None and tile_cache_path is not None:
            raise ValueError(
                "tile-cache state and `tile_cache_path` are mutually exclusive"
            )
        if tile_cache is None and tile_cache_path is None:
            tile_cache_path = (
                Path.cwd() / f"{encoder}_tile_cache_1_3_{tile_size}_{tile_size}"
            )

        network = model_choices[encoder](pretrained=pretrained)
        if device is not None:
            network.to(device)
        self._provided_tile_cache = tile_cache
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        super().__init__(
            network,
            tile_size,
            tile_cache_path,
            defer_prepare=defer_prepare,
            add_keep_modules=[NeighborhoodAttention2D],
            verbose=verbose,
            deterministic=deterministic,
            saliency=saliency,
            diagnose_saliency_assembly=diagnose_saliency_assembly,
            copy_to_gpu=copy_to_gpu,
            statistics_on_cpu=statistics_on_cpu,
            normalize_on_gpu=normalize_on_gpu,
            mean=mean,
            std=std,
        )

    @staticmethod
    def get_model_choices() -> dict[str, Callable[..., nn.Module]]:
        return {
            "nat_mini": nchw_nat_mini,
            "nat_tiny": nchw_nat_tiny,
            "nat_small": nchw_nat_small,
            "nat_base": nchw_nat_base,
            "nat_nano": nchw_nat_nano,
            "nat_pico": nchw_nat_pico,
        }

    @classmethod
    def get_model_names(cls) -> list[str]:
        return list(cls.get_model_choices())

    def load_tile_cache_if_needed(self, use_tile_cache: bool = True):
        if self._provided_tile_cache is not None:
            return self._provided_tile_cache
        return super().load_tile_cache_if_needed(use_tile_cache=use_tile_cache)

    def save_tile_cache_if_needed(self, overwrite: bool = False):
        if self._provided_tile_cache is None:
            return super().save_tile_cache_if_needed(overwrite=overwrite)
        return None


__all__ = ["StreamingNAT"]
