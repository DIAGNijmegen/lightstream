"""Spatial softmax-attention reducers for offline and streaming execution."""

from __future__ import annotations

import torch

from .attention_gem import StreamingAttentionGeMReducer, _normalize_logits
from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import prepare_spatial_mask, resolve_accumulator_dtype


class SoftmaxAttentionReducer(BaseReducer):
    """Average opaque ``values`` using a spatial softmax of attention logits.

    Inputs are positional ``(values, attention_logits)``.  Channel-wise attention
    logits are averaged to one channel before the spatial softmax.
    """

    def __init__(self, accumulator_dtype: torch.dtype | None = None, mask_resize: bool = False,
                 mask_resize_mode: str = "nearest"):
        super().__init__()
        self.accumulator_dtype = accumulator_dtype
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None):
        if len(inputs) != 2:
            raise ValueError(f"SoftmaxAttentionReducer expects exactly two inputs (values, attention_logits), got {len(inputs)}.")
        values, attention_logits = inputs
        if values.ndim != 4:
            raise ValueError(f"Reducer expects NCHW values tensor, got shape={tuple(values.shape)}")
        logits = _normalize_logits(attention_logits, values)
        if self._streaming_passthrough:
            passthrough = (values.view_as(values), logits.view_as(logits))
            self._last_inputs, self._last_output = passthrough, passthrough[0]
            return passthrough

        dtype = resolve_accumulator_dtype(self.accumulator_dtype, values.dtype)
        values_acc, logits_acc = values.to(dtype=dtype), logits.to(dtype=dtype)
        if mask is None:
            valid = torch.ones((values.shape[0], 1, *values.shape[-2:]), device=values.device, dtype=torch.bool)
        else:
            valid = prepare_spatial_mask(mask, values, mask_resize=self.mask_resize,
                                         mask_resize_mode=self.mask_resize_mode)
        masked_logits = torch.where(valid, logits_acc, torch.full_like(logits_acc, torch.finfo(dtype).min))
        maximum = masked_logits.amax(dim=(-2, -1), keepdim=True)
        unnormalized = torch.where(valid, torch.exp(masked_logits - maximum), torch.zeros_like(masked_logits))
        denominator = unnormalized.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        numerator = (unnormalized * values_acc).sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        output = torch.where(denominator > 0, numerator / denominator.clamp_min(torch.finfo(dtype).tiny),
                             torch.zeros_like(numerator))
        return output.to(dtype=values.dtype)

    def to_streaming(self) -> "StreamingSoftmaxAttentionReducer":
        return StreamingSoftmaxAttentionReducer(self.accumulator_dtype, self.mask_resize, self.mask_resize_mode)


class StreamingSoftmaxAttentionReducer(BaseStreamingGlobalReducer):
    """Stable global spatial softmax-attention accumulated across tiles."""

    def __init__(self, accumulator_dtype: torch.dtype | None = None, mask_resize: bool = False,
                 mask_resize_mode: str = "nearest"):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode
        self.register_buffer("running_max", torch.zeros(0), persistent=False)
        self.register_buffer("softmax_denominator", torch.zeros(0), persistent=False)
        self.register_buffer("weighted_numerator", torch.zeros(0), persistent=False)
        self._value_dtype = torch.float32

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None):
        if len(inputs) != 2:
            raise ValueError(f"StreamingSoftmaxAttentionReducer expects exactly two inputs (values, attention_logits), got {len(inputs)}.")
        values, logits = inputs
        _normalize_logits(logits, values)
        self._last_inputs, self._last_output = (values, logits), values
        return values, logits

    # Multi-input tile orchestration and replay bookkeeping are identical to
    # AttentionGeM; only the reduction math below differs.
    accumulate_stream_tile = StreamingAttentionGeMReducer.accumulate_stream_tile
    build_backward_pair = StreamingAttentionGeMReducer.build_backward_pair

    def init_reduction_state(self, *, batch_size, channels, device, dtype, accumulator_dtype):
        self._value_dtype = dtype
        self.running_max = torch.full((batch_size, 1, 1, 1), torch.finfo(accumulator_dtype).min,
                                      device=device, dtype=accumulator_dtype)
        self.softmax_denominator = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=accumulator_dtype)
        self.weighted_numerator = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=accumulator_dtype)

    def accumulate_valid_tile(self, tile, valid_mask: torch.Tensor) -> None:
        values, logits = self._parse_multi_input_payload(tile)
        if self.weighted_numerator.numel() == 0:
            self.reset_stream_state(values.shape[0], values.shape[1], values.device, values.dtype)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, values.dtype)
        values_acc = values.to(dtype=dtype)
        logits_acc = _normalize_logits(logits, values).to(dtype=dtype)
        valid = valid_mask[None, None].to(device=values.device, dtype=torch.bool)
        masked = torch.where(valid, logits_acc, torch.full_like(logits_acc, torch.finfo(dtype).min))
        tile_max = masked.amax(dim=(-2, -1), keepdim=True)
        exponentials = torch.where(valid, torch.exp(masked - tile_max), torch.zeros_like(masked))
        tile_denominator = exponentials.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        tile_numerator = (exponentials * values_acc).sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        new_max = torch.maximum(self.running_max, tile_max)
        old_scale, tile_scale = torch.exp(self.running_max - new_max), torch.exp(tile_max - new_max)
        self.softmax_denominator = self.softmax_denominator * old_scale + tile_denominator * tile_scale
        self.weighted_numerator = self.weighted_numerator * old_scale + tile_numerator * tile_scale
        self.running_max = new_max

    def finalize_from_state(self):
        if self.weighted_numerator.numel() == 0:
            raise RuntimeError("StreamingSoftmaxAttentionReducer state is empty, accumulate_stream_tile() was not called.")
        dtype = self.weighted_numerator.dtype
        result = torch.where(self.softmax_denominator > 0,
                             self.weighted_numerator / self.softmax_denominator.clamp_min(torch.finfo(dtype).tiny),
                             torch.zeros_like(self.weighted_numerator))
        return result.to(dtype=self._value_dtype)

    def extra_state_for_backward(self):
        dtype = self.weighted_numerator.dtype
        denominator = self.softmax_denominator.clamp_min(torch.finfo(dtype).tiny)
        return {"max": self.running_max, "denominator": denominator,
                "mean": self.weighted_numerator / denominator}

    def reduce_tile_for_backward(self, trimmed_output, valid_mask, global_context):
        values, logits = self._parse_multi_input_payload(trimmed_output)
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[-2:], device=values.device, dtype=torch.bool)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, values.dtype)
        values_acc = values.to(dtype=dtype)
        logits_acc = _normalize_logits(logits, values).to(dtype=dtype)
        valid = valid_mask[None, None].to(device=values.device, dtype=torch.bool)
        weights = torch.where(valid, torch.exp(logits_acc - global_context["max"].to(values.device)),
                              torch.zeros_like(logits_acc))
        centered = values_acc - global_context["mean"].to(values.device).detach()
        return streaming_reduce_tile(weights * centered, valid_mask,
                                     global_context["denominator"].to(values.device)).to(values.dtype)

    def to_reducer(self) -> SoftmaxAttentionReducer:
        return SoftmaxAttentionReducer(self.accumulator_dtype, self.mask_resize, self.mask_resize_mode)
