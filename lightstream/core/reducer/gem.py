"""GeM reducer implementations for offline and streaming execution."""

import torch

from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import normalize_spatial_mask, resolve_accumulator_dtype


class GeMReducer(BaseReducer):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Reducer expects NCHW tensor, got shape={tuple(x.shape)}")
        if self._streaming_passthrough:
            return x

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        x_acc = x.to(dtype=acc_dtype)
        x_clamped = x_acc.clamp_min(self.eps)
        r = self.current_r.to(device=x.device, dtype=acc_dtype)
        x_pow = x_clamped.pow(r)

        if mask is not None:
            mask_nchw = normalize_spatial_mask(mask, x)
            mask_acc = mask_nchw.to(dtype=acc_dtype)
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


class StreamingGeMReducer(BaseStreamingGlobalReducer):
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

    def init_reduction_state(self, *, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, accumulator_dtype: torch.dtype) -> None:
        self.running_q = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=accumulator_dtype)

    def accumulate_valid_tile(self, tile: torch.Tensor, valid_mask: torch.Tensor) -> None:
        if self.running_sum.numel() == 0:
            self.reset_stream_state(batch_size=tile.shape[0], channels=tile.shape[1], device=tile.device, dtype=tile.dtype)

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, tile.dtype)
        x = tile.to(dtype=acc_dtype)
        x_clamped = x.clamp_min(self.eps)
        r = self.current_r.to(device=tile.device, dtype=acc_dtype)

        x_pow = x_clamped.pow(r)
        s_tile = streaming_reduce_tile(x_pow, valid_mask, None).to(dtype=self.running_sum.dtype)
        q_tile = streaming_reduce_tile(x_pow * x_clamped.log(), valid_mask, None).to(dtype=self.running_q.dtype)

        self.running_sum = self.running_sum + s_tile
        self.running_q = self.running_q + q_tile

        n_pixels = int(valid_mask.sum().item())
        pixel_increment = torch.tensor(n_pixels, device=self.running_count.device, dtype=self.running_count.dtype)
        self.running_count = self.running_count + pixel_increment

    def finalize_from_state(self) -> torch.Tensor:
        if self.running_sum.numel() == 0:
            raise RuntimeError("StreamingGeMReducer state is empty, accumulate_stream_tile() was not called.")

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, self.running_sum.dtype)
        r = self.current_r.to(device=self.running_sum.device, dtype=acc_dtype)
        s = self.running_sum.to(dtype=acc_dtype)
        n = self.running_count.to(dtype=acc_dtype).clamp_min(1)
        m = (s / n).clamp_min(self.eps)
        y = m.pow(1.0 / r)
        return y.to(dtype=self.running_sum.dtype)

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

    def reduce_tile_for_backward(self, trimmed_output: torch.Tensor, valid_mask: torch.Tensor | None, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        if valid_mask is None:
            raise ValueError("StreamingGeMReducer backward replay requires a valid_mask.")

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, trimmed_output.dtype)
        x = trimmed_output.to(dtype=acc_dtype)
        x_clamped = x.clamp_min(self.eps)
        r = global_context["r"].to(device=trimmed_output.device, dtype=acc_dtype)
        n = global_context["normalization"].to(device=trimmed_output.device, dtype=acc_dtype).clamp_min(1)

        x_pow = x_clamped.pow(r)
        m_tile = streaming_reduce_tile(x_pow, valid_mask, n)
        return m_tile.clamp_min(self.eps).pow(1.0 / r).to(dtype=trimmed_output.dtype)
