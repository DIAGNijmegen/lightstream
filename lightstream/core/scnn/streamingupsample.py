from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingUpsample2d(nn.Module):
    """Streaming-aware wrapper around :class:`torch.nn.Upsample`.

    For now we intentionally keep a narrow, validated surface so we can add
    exact stream-boundary semantics in follow-up iterations.
    """

    _SUPPORTED_MODES = {"nearest", "bilinear"}

    def __init__(
        self,
        size: Optional[tuple[int, int] | int] = None,
        scale_factor: Optional[tuple[float, float] | float] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ):
        super().__init__()

        if size is None and scale_factor is None:
            raise ValueError("StreamingUpsample2d expects either `size` or `scale_factor`.")

        if mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported upsample mode `{mode}` for StreamingUpsample2d. "
                f"Supported modes: {sorted(self._SUPPORTED_MODES)}"
            )

        if mode == "bilinear" and align_corners not in (None, False):
            raise ValueError("StreamingUpsample2d currently supports bilinear mode only with align_corners=False.")

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            inpt,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )

    @classmethod
    def from_torch_upsample(cls, module: nn.Upsample) -> "StreamingUpsample2d":
        return cls(
            size=module.size,
            scale_factor=module.scale_factor,
            mode=module.mode,
            align_corners=module.align_corners,
            recompute_scale_factor=module.recompute_scale_factor,
        )

    def to_torch_upsample(self) -> nn.Upsample:
        return nn.Upsample(
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )
