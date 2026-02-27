import torch
import torch.nn as nn

from torch import Tensor


class GlobalReducer(nn.Module):
    """Generic global reducer: y = post(mean(pointwise(x)))."""

    SUPPORTED_POINTWISE = {"pow", "identity"}
    SUPPORTED_POST = {"pow_inv", "identity"}

    def __init__(self, r: float = 4.0, eps: float = 1e-12, pointwise: str = "pow", post: str = "pow_inv"):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        if pointwise not in self.SUPPORTED_POINTWISE:
            raise ValueError(f"Unsupported pointwise mode '{pointwise}'. Supported: {sorted(self.SUPPORTED_POINTWISE)}")
        if post not in self.SUPPORTED_POST:
            raise ValueError(f"Unsupported post mode '{post}'. Supported: {sorted(self.SUPPORTED_POST)}")

        self.r = float(r)
        self.eps = float(eps)
        self.pointwise = pointwise
        self.post = post

    def _pointwise(self, x: Tensor) -> Tensor:
        if self.pointwise == "pow":
            return x.pow(self.r)
        return x

    def _post(self, mean_phi: Tensor) -> Tensor:
        mean_phi = mean_phi.clamp_min(self.eps)
        if self.post == "pow_inv":
            return mean_phi.pow(1.0 / self.r)
        return mean_phi

    def aggregate(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(x.shape)}")
        mean_phi = self._pointwise(x).mean(dim=(-2, -1), keepdim=True)
        return self._post(mean_phi)

    def forward(self, logits: Tensor) -> Tensor:
        return self.aggregate(logits)
