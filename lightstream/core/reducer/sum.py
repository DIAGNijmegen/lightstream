"""Sum reducer implementations for offline and streaming execution."""

import torch

from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import normalize_spatial_mask, resolve_accumulator_dtype


class SumReducer(BaseReducer):
    """Apply global spatial sum reduction on NCHW tensors."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        super().__init__()
        self.accumulator_dtype = accumulator_dtype

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Reducer expects NCHW tensor, got shape={tuple(x.shape)}")
        if self._streaming_passthrough:
            return x
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        if mask is not None:
            mask_nchw = normalize_spatial_mask(mask, x)
            masked = x * mask_nchw.to(dtype=x.dtype)
            return masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)
        return x.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        return StreamingSumReducer(accumulator_dtype=self.accumulator_dtype)


class StreamingSumReducer(BaseStreamingGlobalReducer):
    """Streaming reducer configured for sum semantics."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        super().__init__(mode="sum", accumulator_dtype=accumulator_dtype)

    def init_reduction_state(self, *, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, accumulator_dtype: torch.dtype) -> None:
        _ = (batch_size, channels, device, dtype, accumulator_dtype)

    def accumulate_valid_tile(self, tile: torch.Tensor, valid_mask: torch.Tensor) -> None:
        if self.running_sum.numel() == 0:
            self.reset_stream_state(batch_size=tile.shape[0], channels=tile.shape[1], device=tile.device, dtype=tile.dtype)
        tile_contribution = streaming_reduce_tile(tile, valid_mask, None)
        self.running_sum = self.running_sum + tile_contribution

    def finalize_from_state(self) -> torch.Tensor:
        if self.running_sum.numel() == 0:
            raise RuntimeError("StreamingReducer state is empty, accumulate_stream_tile() was not called.")
        return self.running_sum

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        return {"normalization": None}

    def reduce_tile_for_backward(self, trimmed_output: torch.Tensor, valid_mask: torch.Tensor | None, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        return streaming_reduce_tile(trimmed_output, valid_mask, global_context.get("normalization"))
