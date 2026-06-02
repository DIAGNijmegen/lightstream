"""Configuration container for the streaming engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StreamingConfig:
    tile_shape: tuple
    verbose: bool = False
    deterministic: bool = False
    saliency: bool = False
    eps: float = 1e-5
    copy_to_gpu: bool = True
    dtype: Any = None
    statistics_on_cpu: bool = False
    normalize_on_gpu: bool = False
    mean: Any = None
    std: Any = None
