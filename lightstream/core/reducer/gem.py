"""GeM reducer implementations for offline and streaming execution."""

from dataclasses import dataclass

import torch

from .base import BaseStreamingGlobalReducer, ReducerMeta, ReducerTile, streaming_reduce_tile
from .reducer_base import ManualVJPReducer, SpatialReducer
from .utils import resolve_accumulator_dtype


class GeMReducer(SpatialReducer):
    """Apply global generalized-mean (GeM) reduction on NCHW tensors."""

    def __init__(
        self,
        r_init: float = 4.0,
        eps: float = 1e-6,
        accumulator_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = float(eps)
        self.accumulator_dtype = accumulator_dtype

        init_r = torch.tensor(float(r_init), dtype=torch.float32)
        self.register_buffer("r", init_r)

    @property
    def current_r(self) -> torch.Tensor:
        return self.r

    def reduce_spatial(self, x: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        x_acc = x.to(dtype=acc_dtype)
        x_clamped = x_acc.clamp_min(self.eps)
        r = self.current_r.to(device=x.device, dtype=acc_dtype)
        x_pow = x_clamped.pow(r)

        if mask is not None:
            mask_acc = mask.to(dtype=acc_dtype)
            denom = mask_acc.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).clamp_min(1)
            mean_pow = (x_pow * mask_acc).sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype) / denom
        else:
            mean_pow = x_pow.mean(dim=(-2, -1), keepdim=True, dtype=acc_dtype)

        y = mean_pow.clamp_min(self.eps).pow(1.0 / r)
        return y.to(dtype=x.dtype)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        reducer = StreamingGeMReducer(
            r_init=float(self.current_r.detach().item()),
            eps=self.eps,
            accumulator_dtype=self.accumulator_dtype,
        )
        reducer.r.data.copy_(self.current_r.detach().to(device=reducer.r.device, dtype=reducer.r.dtype))
        return reducer


@dataclass
class GeMState:
    running_sum: torch.Tensor
    running_count: torch.Tensor
    running_q: torch.Tensor


class StreamingGeMReducer(ManualVJPReducer):
    """Streaming global GeM reducer."""

    def __init__(
        self,
        r_init: float = 4.0,
        eps: float = 1e-6,
        accumulator_dtype: torch.dtype | None = None,
    ):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
        self.eps = float(eps)
        init_r = torch.tensor(float(r_init), dtype=torch.float32)
        self.register_buffer("r", init_r)

        self.register_buffer("running_q", torch.zeros(0), persistent=False)

    @property
    def current_r(self) -> torch.Tensor:
        return self.r

    def init_state(self, meta: ReducerMeta) -> "GeMState":
        running_sum = torch.zeros((meta.batch_size, meta.channels, 1, 1), device=meta.device, dtype=meta.dtype)
        running_count = torch.zeros((meta.batch_size, 1, 1, 1), device=meta.device, dtype=meta.accumulator_dtype)
        running_q = torch.zeros((meta.batch_size, meta.channels, 1, 1), device=meta.device, dtype=meta.dtype)
        self.running_sum = running_sum
        self.running_count = running_count
        self.running_q = running_q
        return GeMState(running_sum=running_sum, running_count=running_count, running_q=running_q)

    def update(self, state: "GeMState", tile: ReducerTile) -> "GeMState":
        (x_tile,) = tile.tensors
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x_tile.dtype)
        x = x_tile.to(dtype=acc_dtype)
        x_clamped = x.clamp_min(self.eps)
        r = self.current_r.to(device=x_tile.device, dtype=acc_dtype)

        x_pow = x_clamped.pow(r)
        s_tile = streaming_reduce_tile(x_pow, tile.mask, None).to(dtype=state.running_sum.dtype)
        q_tile = streaming_reduce_tile(x_pow * x_clamped.log(), tile.mask, None).to(dtype=state.running_q.dtype)

        state.running_sum = state.running_sum + s_tile
        state.running_q = state.running_q + q_tile
        n_pixels = int(tile.mask.sum().item()) if tile.mask is not None else int(x_tile.shape[-2] * x_tile.shape[-1])
        state.running_count = state.running_count + torch.tensor(n_pixels, device=state.running_count.device, dtype=state.running_count.dtype)
        self.running_sum = state.running_sum
        self.running_count = state.running_count
        self.running_q = state.running_q
        return state

    def finalize(self, state: "GeMState") -> torch.Tensor:
        if state.running_sum.numel() == 0:
            raise RuntimeError("StreamingGeMReducer state is empty.")

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, state.running_sum.dtype)
        r = self.current_r.to(device=state.running_sum.device, dtype=acc_dtype)
        s = state.running_sum.to(dtype=acc_dtype)
        n = state.running_count.to(dtype=acc_dtype).clamp_min(1)
        m = (s / n).clamp_min(self.eps)
        y = m.pow(1.0 / r)
        return y.to(dtype=state.running_sum.dtype)

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, self.running_sum.dtype)
        r = self.current_r.to(device=self.running_sum.device, dtype=acc_dtype)
        n = self.running_count.to(dtype=acc_dtype).clamp_min(1)
        s = self.running_sum.to(dtype=acc_dtype)
        m = (s / n).clamp_min(self.eps)
        return {
            "normalization": n,
            "r": r,
            "m": m,
            "q": self.running_q.to(dtype=acc_dtype),
        }

    def reduce_tile_for_backward(self, tile: ReducerTile, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        if tile.mask is None:
            raise ValueError("StreamingGeMReducer backward replay requires a tile mask.")
        trimmed_output = tile.tensors[0]

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, trimmed_output.dtype)
        x = trimmed_output.to(dtype=acc_dtype)
        x_clamped = x.clamp_min(self.eps)
        r = global_context["r"].to(device=trimmed_output.device, dtype=acc_dtype)
        n = global_context["normalization"].to(device=trimmed_output.device, dtype=acc_dtype).clamp_min(1)

        x_pow = x_clamped.pow(r)
        local_m = streaming_reduce_tile(x_pow, tile.mask, n)

        global_m = global_context["m"].to(device=trimmed_output.device, dtype=acc_dtype)
        scale = (1.0 / r) * global_m.clamp_min(self.eps).pow(1.0 / r - 1.0)
        return (scale.detach() * local_m).to(dtype=trimmed_output.dtype)

    def to_reducer(self) -> GeMReducer:
        reducer = GeMReducer(
            r_init=float(self.current_r.detach().item()),
            eps=self.eps,
            accumulator_dtype=self.accumulator_dtype,
        )
        reducer.r.data.copy_(self.current_r.detach().to(device=reducer.r.device, dtype=reducer.r.dtype))
        return reducer
