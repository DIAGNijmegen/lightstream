"""Size-focal pooling reducers."""

from __future__ import annotations

import math
import torch

from .base import BaseStreamingGlobalReducer
from .reducer_base import BaseReducer
from .utils import prepare_spatial_mask, resolve_accumulator_dtype


def _validate_m(m: torch.Tensor) -> None:
    if not isinstance(m, torch.Tensor):
        raise TypeError("SizeFocalReducer input m must be a tensor.")
    if m.ndim != 4:
        raise ValueError(f"SizeFocalReducer expects an NCHW [N, C, H, W] tensor, got {tuple(m.shape)}.")


def _validate_parameters(p: float, lambda_: float) -> tuple[float, float]:
    if not math.isfinite(p):
        raise ValueError("p must be finite.")
    if not math.isfinite(lambda_) or lambda_ <= 0:
        raise ValueError("lambda_ must be positive and finite.")
    return float(p), float(lambda_)


class SizeFocalReducer(BaseReducer):
    """Compute size-focal scores from one activation/probability map."""

    def __init__(self, p: float = 1.0, lambda_: float = 1e-6, accumulator_dtype: torch.dtype | None = None,
                 mask_resize: bool = False, mask_resize_mode: str = "nearest"):
        super().__init__()
        self.p, self.lambda_ = _validate_parameters(p, lambda_)
        self.accumulator_dtype, self.mask_resize, self.mask_resize_mode = accumulator_dtype, bool(mask_resize), mask_resize_mode

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if len(inputs) != 1: raise ValueError(f"SizeFocalReducer expects exactly one input, got {len(inputs)}.")
        m = inputs[0]; _validate_m(m)
        if self._streaming_passthrough: return m
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, m.dtype)
        value = m.to(dtype)
        if mask is None:
            count = torch.full((m.shape[0], 1, 1, 1), m.shape[-2] * m.shape[-1], device=m.device, dtype=dtype)
        else:
            tissue = prepare_spatial_mask(mask, m, mask_resize=self.mask_resize, mask_resize_mode=self.mask_resize_mode).to(dtype)
            value = value * tissue
            count = tissue.sum((-2, -1), keepdim=True, dtype=dtype)
        mean = value.sum((-2, -1), keepdim=True, dtype=dtype) / count.clamp_min(1)
        result = (1 - mean).pow(self.p) * torch.log(mean + self.lambda_)
        return torch.where(count > 0, result, torch.zeros_like(result)).to(m.dtype)

    def to_streaming(self):
        return StreamingSizeFocalReducer(self.p, self.lambda_, self.accumulator_dtype, self.mask_resize, self.mask_resize_mode)


class StreamingSizeFocalReducer(BaseStreamingGlobalReducer):
    """Streaming size-focal reducer; nonlinear finalization happens once globally."""

    def __init__(self, p: float = 1.0, lambda_: float = 1e-6, accumulator_dtype: torch.dtype | None = None,
                 mask_resize: bool = False, mask_resize_mode: str = "nearest"):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
        self.p, self.lambda_ = _validate_parameters(p, lambda_)
        self.mask_resize, self.mask_resize_mode = bool(mask_resize), mask_resize_mode

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None):
        if len(inputs) != 1:
            raise ValueError(f"StreamingSizeFocalReducer expects exactly one input, got {len(inputs)}.")
        _validate_m(inputs[0])
        self._last_output = inputs[0]
        return inputs[0]

    def init_reduction_state(self, *, batch_size, channels, device, dtype, accumulator_dtype):
        self._output_dtype = dtype
        self.running_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=accumulator_dtype)

    def accumulate_valid_tile(self, tile, valid_mask):
        _validate_m(tile)
        if self.running_sum.numel() == 0: self.reset_stream_state(tile.shape[0], tile.shape[1], tile.device, tile.dtype)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, tile.dtype)
        valid = valid_mask[None, None].to(device=tile.device, dtype=dtype)
        self.running_sum = self.running_sum + (tile.to(dtype) * valid).sum((-2, -1), keepdim=True, dtype=dtype)
        self.running_count = self.running_count + valid.sum((-2, -1), keepdim=True, dtype=dtype)

    def finalize_from_state(self):
        if self.running_sum.numel() == 0: raise RuntimeError("StreamingSizeFocalReducer state is empty, accumulate_stream_tile() was not called.")
        count = self.running_count.to(self.running_sum.dtype)
        mean = self.running_sum / count.clamp_min(1)
        result = (1 - mean).pow(self.p) * torch.log(mean + self.lambda_)
        return torch.where(count > 0, result, torch.zeros_like(result)).to(self._output_dtype)

    def extra_state_for_backward(self):
        count = self.running_count
        mean = self.running_sum / count.to(self.running_sum.dtype).clamp_min(1)
        # d/dmean [(1-mean)^p log(lambda + mean)]
        derivative = (-self.p * (1 - mean).pow(self.p - 1) * torch.log(mean + self.lambda_)
                      + (1 - mean).pow(self.p) / (mean + self.lambda_))
        return {"count": count, "derivative": torch.where(count > 0, derivative, torch.zeros_like(derivative))}

    def reduce_tile_for_backward(self, trimmed_output, valid_mask, global_context):
        _validate_m(trimmed_output)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, trimmed_output.dtype)
        valid = 1 if valid_mask is None else valid_mask[None, None].to(device=trimmed_output.device, dtype=dtype)
        local_sum = (trimmed_output.to(dtype) * valid).sum((-2, -1), keepdim=True, dtype=dtype)
        return (local_sum * global_context["derivative"].to(device=trimmed_output.device, dtype=dtype)
                / global_context["count"].to(device=trimmed_output.device, dtype=dtype).clamp_min(1)).to(trimmed_output.dtype)

    def to_reducer(self):
        return SizeFocalReducer(self.p, self.lambda_, self.accumulator_dtype, self.mask_resize, self.mask_resize_mode)
