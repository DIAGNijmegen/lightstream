from __future__ import annotations

import torch
import torch.nn as nn


class StreamingGlobalReducer(nn.Module):
    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        self.r = float(r)
        self.eps = float(eps)
        self.register_buffer("sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("count", torch.tensor(0.0), persistent=False)
        self.grad_lost = None
        self.output_stride = torch.tensor([1, 1, 1])
        self.reset()

    def reset(self):
        self.input_loc = None
        self.sum = self.sum.new_zeros((1, 1))
        self.count = self.count.new_zeros((1, 1))

    def _ensure_state(self, x: torch.Tensor):
        if (
            self.sum.device != x.device
            or self.sum.dtype != x.dtype
            or self.sum.ndim != 2
            or self.sum.shape[0] != x.shape[0]
            or self.sum.shape[1] != x.shape[1]
        ):
            self.sum = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
            self.count = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(x.shape)}")

        reduced = x.pow(self.r)
        self._ensure_state(reduced)

        tile_sum = reduced.sum(dim=(-2, -1))
        tile_count = torch.full_like(tile_sum, float(reduced.shape[-2] * reduced.shape[-1]))

        self.sum = self.sum + tile_sum
        self.count = self.count + tile_count

        mean_p_r = self.sum / self.count.clamp_min(1.0)
        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)[:, :, None, None]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.aggregate(logits)

    @classmethod
    def from_global_reducer(cls, module: nn.Module) -> "StreamingGlobalReducer":
        return cls(r=float(module.r), eps=float(module.eps))

    def to_global_reducer(self) -> nn.Module:
        from lightstream.models.segment.globalreducer import GlobalReducer

        return GlobalReducer(r=self.r, eps=self.eps)
