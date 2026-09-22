"""Public streamed Neighborhood Attention Transformer feature extractors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lightstream.core.layers import NeighborhoodAttention2D
import lightstream.models.nat.nchw as nchw
from lightstream.modules.streaming import StreamingModule

_FACTORY_PREFIX = "nchw_nat_"


def _factory_name(variant: str) -> str:
    """Return the conventional NCHW factory name for a public variant name."""

    if not isinstance(variant, str) or not variant:
        raise ValueError("NAT variant must be a non-empty string")
    name = variant if variant.startswith("nat_") else f"nat_{variant}"
    suffix = name.removeprefix("nat_")
    if not suffix or not suffix.isidentifier():
        raise ValueError(
            f"Invalid NAT variant {variant!r}; expected 'nat_<name>' or '<name>'"
        )
    return f"nchw_{name}"


def _resolve_factory(variant: str):
    """Resolve and validate an exported ``nchw_nat_*`` model factory."""

    factory_name = _factory_name(variant)
    exported = getattr(nchw, "__all__", ())
    factory = getattr(nchw, factory_name, None)
    if (
        not factory_name.startswith(_FACTORY_PREFIX)
        or factory_name not in exported
        or not callable(factory)
    ):
        choices = ", ".join(StreamingNAT.get_model_names())
        raise ValueError(f"Invalid NAT variant {variant!r}. Choose one of: {choices}")
    return factory


class StreamingNAT(StreamingModule):
    """Stream a deterministic NAT backbone and return its normalized NCHW map.

    ``variant`` accepts either the public name (for example ``"nat_mini"``)
    or its short form (``"mini"``).  Checkpoint selection and conversion are
    delegated to the corresponding NCHW factory.
    """

    def __init__(
        self,
        variant: str,
        tile_size: int,
        *,
        pretrained: Any = True,
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
        factory = _resolve_factory(variant)
        stochastic = {
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "drop_path_rate": drop_path_rate,
        }
        invalid = [
            f"{name}={value!r}" for name, value in stochastic.items() if value != 0
        ]
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
            raise ValueError(
                "tile-cache state and `tile_cache_path` are mutually exclusive"
            )
        if tile_cache is None and tile_cache_path is None:
            model_name = factory.__name__.removeprefix("nchw_")
            tile_cache_path = Path.cwd() / (
                f"{model_name}_tile_cache_1_3_{tile_size}_{tile_size}"
            )

        network = factory(pretrained=pretrained)
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
    def get_model_names() -> list[str]:
        """Discover public variants from exported NCHW factory callables."""

        return sorted(
            name.removeprefix("nchw_")
            for name in getattr(nchw, "__all__", ())
            if name.startswith(_FACTORY_PREFIX) and callable(getattr(nchw, name, None))
        )

    def load_tile_cache_if_needed(self, use_tile_cache: bool = True):
        if self._provided_tile_cache is not None:
            return self._provided_tile_cache
        return super().load_tile_cache_if_needed(use_tile_cache=use_tile_cache)

    def save_tile_cache_if_needed(self, overwrite: bool = False):
        if self._provided_tile_cache is None:
            return super().save_tile_cache_if_needed(overwrite=overwrite)
        return None


__all__ = ["StreamingNAT"]
