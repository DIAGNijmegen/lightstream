import torch
import torch.nn as nn

from .base import StreamingReducer
from .utils import normalize_spatial_mask, resolve_accumulator_dtype


class Reducer(nn.Module):
    """Apply global spatial sum or mean reduction on NCHW tensors.

    Parameters
    ----------
    mode : str, default="mean"
        Reduction mode, either ``"sum"`` or ``"mean"``.
    accumulator_dtype : torch.dtype | None, default=None
        Optional accumulator dtype for reduction math.
    """

    def __init__(self, mode: str = "mean", accumulator_dtype: torch.dtype | None = None):
        super().__init__()
        if mode not in {"sum", "mean"}:
            raise ValueError(f"Unsupported reducer mode '{mode}', expected 'sum' or 'mean'.")
        self.mode = mode
        self.accumulator_dtype = accumulator_dtype
        self._streaming_passthrough = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reduce spatial dimensions for a batch of feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``[N, C, H, W]``.
        mask : torch.Tensor | None, default=None
            Optional spatial mask (2D/3D/4D) to limit contributing pixels.

        Returns
        -------
        torch.Tensor
            Reduced output with shape ``[N, C, 1, 1]``.
        """
        if x.ndim != 4:
            raise ValueError(f"Reducer expects NCHW tensor, got shape={tuple(x.shape)}")
        if self._streaming_passthrough:
            return x
        if mask is not None:
            mask_nchw = normalize_spatial_mask(mask, x)
            masked = x * mask_nchw.to(dtype=x.dtype)
            acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
            if self.mode == "sum":
                return masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)
            denom = mask_nchw.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).clamp_min(1)
            mean = masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype) / denom
            return mean.to(dtype=x.dtype)
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        if self.mode == "sum":
            return x.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)
        return x.mean(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)


class StreamingMeanReducer(StreamingReducer):
    """Streaming reducer configured for mean semantics."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        """Create a mean-mode streaming reducer.

        Parameters
        ----------
        accumulator_dtype : torch.dtype | None, default=None
            Optional accumulator dtype for running count normalization.
        """
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
