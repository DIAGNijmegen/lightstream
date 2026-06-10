from __future__ import annotations

import torch
import torch.nn as nn


class ChannelLayerNorm(nn.Module):
    """Apply layer normalization across channels for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ChannelLayerNorm expects 4D NCHW input, got {x.ndim}D input with shape {tuple(x.shape)}.")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"ChannelLayerNorm expected {self.num_channels} channels at dimension 1, "
                f"got {x.shape[1]} channels for input shape {tuple(x.shape)}."
            )

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)
