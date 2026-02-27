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

    def aggregate(self, x: Tensor) -> Tensor:
        """
        Aggregation function
        
        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, C, H, W).

        Returns
        -------
        y: torch.Tensor
            A tensor of shape (N, C, 1, 1). Spatial dimensions are not reduced for streaming compatibility
        """
        if x.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(x.shape)}")
        mean_p_r = x.pow(self.r).mean(dim=(-2, -1),keepdim=True)
        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)

    def forward(self, logits: Tensor) -> Tensor:
        return self.aggregate(logits)

if __name__ == "__main__":
    inputs = torch.randn(2, 3, 64, 64)
    print(inputs.shape)
    reducer = GlobalReducer()
    out = reducer(inputs)
    print(out.shape)
