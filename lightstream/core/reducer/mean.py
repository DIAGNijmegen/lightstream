"""Mean reducer implementations for offline and streaming execution."""

import torch

from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .sum import StreamingSumReducer
from .utils import prepare_spatial_mask, resolve_accumulator_dtype


class MeanReducer(BaseReducer):
    """Apply global spatial mean reduction on NCHW tensors."""

    def __init__(
        self,
        accumulator_dtype: torch.dtype | None = None,
        mask_resize: bool = False,
        mask_resize_mode: str = "nearest",
    ):
        super().__init__()
        self.accumulator_dtype = accumulator_dtype
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if len(inputs) != 1:
            raise ValueError(f"MeanReducer expects exactly one tensor input, got {len(inputs)}.")
        x = inputs[0]
        if x.ndim != 4:
            raise ValueError(f"Reducer expects NCHW tensor, got shape={tuple(x.shape)}")
        if self._streaming_passthrough:
            return x
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        if mask is not None:
            mask_nchw = prepare_spatial_mask(mask, x, mask_resize=self.mask_resize, mask_resize_mode=self.mask_resize_mode)
            masked = x * mask_nchw.to(dtype=x.dtype)
            denom = mask_nchw.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).clamp_min(1)
            mean = masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype) / denom
            return mean.to(dtype=x.dtype)
        return x.mean(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        return StreamingMeanReducer(accumulator_dtype=self.accumulator_dtype)


class StreamingMeanReducer(StreamingSumReducer):
    """Streaming reducer configured for mean semantics."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        super().__init__(accumulator_dtype=accumulator_dtype)
        self.mode = "mean"

    def accumulate_valid_tile(self, tile: torch.Tensor, valid_mask: torch.Tensor) -> None:
        super().accumulate_valid_tile(tile, valid_mask)
        n_pixels = int(valid_mask.sum().item())
        pixel_increment = torch.tensor(n_pixels, device=self.running_count.device, dtype=self.running_count.dtype)
        self.running_count = self.running_count + pixel_increment

    def finalize_from_state(self) -> torch.Tensor:
        if self.running_sum.numel() == 0:
            raise RuntimeError("StreamingReducer state is empty, accumulate_stream_tile() was not called.")
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, self.running_sum.dtype)
        denom = self.running_count.to(dtype=acc_dtype).clamp_min(1)
        if denom.dtype != self.running_sum.dtype:
            denom = denom.to(dtype=self.running_sum.dtype)
        return self.running_sum / denom

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        return {"normalization": self.running_count}

    def reduce_tile_for_backward(self, trimmed_output: torch.Tensor, valid_mask: torch.Tensor | None, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        return streaming_reduce_tile(trimmed_output, valid_mask, global_context.get("normalization"))

    def to_reducer(self) -> MeanReducer:
        return MeanReducer(accumulator_dtype=self.accumulator_dtype)
