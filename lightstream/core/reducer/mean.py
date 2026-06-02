"""Mean reducer implementations for offline and streaming execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import BaseStreamingGlobalReducer, ReducerMeta, ReducerTile, streaming_reduce_tile
from .reducer_base import SpatialReducer
from .utils import normalize_spatial_mask, resolve_accumulator_dtype


class MeanReducer(SpatialReducer):
    """Apply global spatial mean reduction on NCHW tensors."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        super().__init__()
        self.accumulator_dtype = accumulator_dtype

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
            mask_nchw = normalize_spatial_mask(mask, x)
            masked = x * mask_nchw.to(dtype=x.dtype)
            denom = mask_nchw.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).clamp_min(1)
            mean = masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype) / denom
            return mean.to(dtype=x.dtype)
        return x.mean(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        return StreamingMeanReducer(accumulator_dtype=self.accumulator_dtype)


@dataclass
class MeanState:
    running_sum: torch.Tensor
    running_count: torch.Tensor


class StreamingMeanReducer(BaseStreamingGlobalReducer):
    """Streaming reducer configured for mean semantics using the ReducerTile API."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)

    def init_state(self, meta: ReducerMeta) -> MeanState:
        running_sum = torch.zeros((meta.batch_size, meta.channels, 1, 1), device=meta.device, dtype=meta.dtype)
        running_count = torch.zeros((meta.batch_size, 1, 1, 1), device=meta.device, dtype=meta.dtype)
        self.running_sum = running_sum
        self.running_count = running_count
        return MeanState(running_sum=running_sum, running_count=running_count)

    def update(self, state: MeanState, tile: ReducerTile) -> MeanState:
        (x,) = tile.tensors
        tile_contribution = streaming_reduce_tile(x, tile.mask, None).to(dtype=state.running_sum.dtype)
        state.running_sum = state.running_sum + tile_contribution
        n_pixels = int(tile.mask.sum().item()) if tile.mask is not None else int(x.shape[-2] * x.shape[-1])
        state.running_count = state.running_count + torch.tensor(n_pixels, device=state.running_count.device, dtype=state.running_count.dtype)
        self.running_sum = state.running_sum
        self.running_count = state.running_count
        return state

    def finalize(self, state: MeanState) -> torch.Tensor:
        if state.running_sum.numel() == 0:
            raise RuntimeError("StreamingMeanReducer state is empty.")
        denom = state.running_count.to(dtype=state.running_sum.dtype).clamp_min(1)
        return state.running_sum / denom

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        return {"normalization": self.running_count}

    def reduce_tile_for_backward(self, tile: ReducerTile, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        return streaming_reduce_tile(tile.tensors[0], tile.mask, global_context.get("normalization"))

    def to_reducer(self) -> MeanReducer:
        return MeanReducer(accumulator_dtype=self.accumulator_dtype)
