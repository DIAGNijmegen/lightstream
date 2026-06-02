"""Sum reducer implementations for offline and streaming execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import BaseStreamingGlobalReducer, ReducerMeta, ReducerTile, streaming_reduce_tile
from .reducer_base import ManualVJPReducer, SpatialReducer
from .utils import normalize_spatial_mask, resolve_accumulator_dtype


class SumReducer(SpatialReducer):
    """Apply global spatial sum reduction on NCHW tensors."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        super().__init__()
        self.accumulator_dtype = accumulator_dtype

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if len(inputs) != 1:
            raise ValueError(f"SumReducer expects exactly one tensor input, got {len(inputs)}.")
        x = inputs[0]
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


@dataclass
class SumState:
    running_sum: torch.Tensor


class StreamingSumReducer(ManualVJPReducer):
    """Streaming reducer configured for sum semantics using the ReducerTile API."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        super().__init__(mode="sum", accumulator_dtype=accumulator_dtype)

    def init_state(self, meta: ReducerMeta) -> SumState:
        running_sum = torch.zeros((meta.batch_size, meta.channels, 1, 1), device=meta.device, dtype=meta.dtype)
        self.running_sum = running_sum
        return SumState(running_sum=running_sum)

    def update(self, state: SumState, tile: ReducerTile) -> SumState:
        (x,) = tile.tensors
        tile_contribution = streaming_reduce_tile(x, tile.mask, None).to(dtype=state.running_sum.dtype)
        state.running_sum = state.running_sum + tile_contribution
        self.running_sum = state.running_sum
        return state

    def finalize(self, state: SumState) -> torch.Tensor:
        if state.running_sum.numel() == 0:
            raise RuntimeError("StreamingSumReducer state is empty.")
        return state.running_sum

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        return {"normalization": None}

    def reduce_tile_for_backward(self, tile: ReducerTile, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        return streaming_reduce_tile(tile.tensors[0], tile.mask, global_context.get("normalization"))

    def to_reducer(self) -> SumReducer:
        return SumReducer(accumulator_dtype=self.accumulator_dtype)
