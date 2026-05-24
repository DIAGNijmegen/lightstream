"""Mean/sum reducer implementations for offline and streaming execution."""

import torch

from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import normalize_spatial_mask, resolve_accumulator_dtype


class Reducer(BaseReducer):
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

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        """Create streaming reducer with equivalent sum/mean semantics."""
        if self.mode == "sum":
            return StreamingSumReducer(accumulator_dtype=self.accumulator_dtype)
        return StreamingMeanReducer(accumulator_dtype=self.accumulator_dtype)


class _StreamingSumMeanReducer(BaseStreamingGlobalReducer):
    """Concrete streaming implementation for sum/mean reductions."""

    def init_reduction_state(self, *, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, accumulator_dtype: torch.dtype) -> None:
        _ = (batch_size, channels, device, dtype, accumulator_dtype)

    def accumulate_valid_tile(self, tile: torch.Tensor, valid_mask: torch.Tensor) -> None:
        if self.running_sum.numel() == 0:
            self.reset_stream_state(batch_size=tile.shape[0], channels=tile.shape[1], device=tile.device, dtype=tile.dtype)
        tile_contribution = streaming_reduce_tile(tile, valid_mask, None)
        self.running_sum = self.running_sum + tile_contribution
        if self.mode == "mean":
            n_pixels = int(valid_mask.sum().item())
            pixel_increment = torch.tensor(n_pixels, device=self.running_count.device, dtype=self.running_count.dtype)
            self.running_count = self.running_count + pixel_increment

    def finalize_from_state(self) -> torch.Tensor:
        if self.running_sum.numel() == 0:
            raise RuntimeError("StreamingReducer state is empty, accumulate_stream_tile() was not called.")
        if self.mode == "sum":
            return self.running_sum
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, self.running_sum.dtype)
        denom = self.running_count.to(dtype=acc_dtype).clamp_min(1)
        if denom.dtype != self.running_sum.dtype:
            denom = denom.to(dtype=self.running_sum.dtype)
        return self.running_sum / denom

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        return {"normalization": self.running_count if self.mode == "mean" else None}

    def reduce_tile_for_backward(self, trimmed_output: torch.Tensor, valid_mask: torch.Tensor | None, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        return streaming_reduce_tile(trimmed_output, valid_mask, global_context.get("normalization"))


class StreamingMeanReducer(_StreamingSumMeanReducer):
    """Streaming reducer configured for mean semantics."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        """Create a mean-mode streaming reducer.

        Parameters
        ----------
        accumulator_dtype : torch.dtype | None, default=None
            Optional accumulator dtype for running count normalization.
        """
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)


class StreamingSumReducer(_StreamingSumMeanReducer):
    """Streaming reducer configured for sum semantics."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        super().__init__(mode="sum", accumulator_dtype=accumulator_dtype)
