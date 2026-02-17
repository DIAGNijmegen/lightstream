import torch
import torch.nn as nn

from torch import Tensor


class GlobalReducer(nn.Module):
    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        self.r = float(r)
        self.eps = float(eps)

    def aggregate(self, logits: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(logits.shape)}")
        probs = torch.sigmoid(logits)
        mean_p_r = probs.pow(self.r).mean(dim=(-2, -1))
        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)

    def forward(self, logits: Tensor) -> Tensor:
        return self.aggregate(logits)
