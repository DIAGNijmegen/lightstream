"""Normalized-sigmoid attention reduction, offline and streaming."""

from __future__ import annotations

import torch

from .attention_gem import _normalize_logits
from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import prepare_spatial_mask, resolve_accumulator_dtype


class NormalizedSigmoidAttentionReducer(BaseReducer):
    """Average raw ``values`` using spatially normalized sigmoid attention.

    Channel-shaped attention logits are averaged to one channel, matching
    :class:`AttentionGeMReducer`, and that shared attention field is broadcast
    over all value channels.
    """

    def __init__(self, accumulator_dtype=None, mask_resize=False, mask_resize_mode="nearest"):
        super().__init__()
        self.accumulator_dtype = accumulator_dtype
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode

    def forward(self, *inputs, mask=None):
        if len(inputs) != 2:
            raise ValueError(
                "NormalizedSigmoidAttentionReducer expects exactly two inputs "
                f"(values, attention_logits), got {len(inputs)}."
            )
        values, attention_logits = inputs
        if values.ndim != 4:
            raise ValueError(f"Reducer expects NCHW values, got shape={tuple(values.shape)}")
        logits = _normalize_logits(attention_logits, values)
        if self._streaming_passthrough:
            payload = (values.view_as(values), logits.view_as(logits))
            self._last_inputs, self._last_output = payload, payload[0]
            return payload

        dtype = resolve_accumulator_dtype(self.accumulator_dtype, values.dtype)
        values_acc = values.to(dtype)
        scores = torch.sigmoid(logits.to(dtype))
        if mask is not None:
            valid = prepare_spatial_mask(
                mask, values, mask_resize=self.mask_resize,
                mask_resize_mode=self.mask_resize_mode,
            )
            scores = torch.where(valid, scores, torch.zeros_like(scores))
        numerator = (scores * values_acc).sum((-2, -1), keepdim=True, dtype=dtype)
        denominator = scores.sum((-2, -1), keepdim=True, dtype=dtype)
        safe = denominator.clamp_min(torch.finfo(dtype).tiny)
        result = torch.where(denominator > 0, numerator / safe, torch.zeros_like(numerator))
        return result.to(values.dtype)

    def to_streaming(self):
        return StreamingNormalizedSigmoidAttentionReducer(
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
        )


class StreamingNormalizedSigmoidAttentionReducer(BaseStreamingGlobalReducer):
    """Tiled execution implementation of normalized-sigmoid attention."""

    def __init__(self, accumulator_dtype=None, mask_resize=False, mask_resize_mode="nearest"):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode
        self.register_buffer("running_numerator", torch.zeros(0), persistent=False)
        self.register_buffer("running_denominator", torch.zeros(0), persistent=False)

    def forward(self, *inputs, mask=None):
        if len(inputs) != 2:
            raise ValueError(
                "StreamingNormalizedSigmoidAttentionReducer expects exactly two inputs "
                f"(values, attention_logits), got {len(inputs)}."
            )
        values, logits = inputs
        if values.ndim != 4:
            raise ValueError(f"Reducer expects NCHW values, got shape={tuple(values.shape)}")
        _normalize_logits(logits, values)
        self._last_inputs, self._last_output = (values, logits), values
        return values, logits

    def accumulate_stream_tile(self, trimmed_output, tile_y, tile_x, sides, dst_box, user_mask=None):
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 2:
            raise ValueError(f"StreamingNormalizedSigmoidAttentionReducer expects payload arity=2, got {len(payload)}")
        values = payload[0]
        y0, y1, x0, x1 = dst_box
        seen = self._stream_seen_mask[y0:y1, x0:x1]
        new = ~seen
        effective = new if user_mask is None else new & user_mask.to(device=new.device, dtype=torch.bool)
        if self._debug_replay_enabled:
            self._replay_assignments.append((int(tile_y), int(tile_x), bool(sides.top), bool(sides.left), bool(sides.right), bool(sides.bottom), int(values.shape[-2]), int(values.shape[-1]), int(y0), int(y1), int(x0), int(x1), 2))
        if torch.any(effective):
            self.accumulate_valid_tile(payload, effective)
        seen |= new

    def build_backward_pair(self, trimmed_output, gradient, *, input_y, input_x, sides, valid_mask=None):
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 2:
            raise ValueError(f"StreamingNormalizedSigmoidAttentionReducer expects payload arity=2, got {len(payload)}")
        values = payload[0]
        if self._debug_replay_enabled:
            self._replay_cursor = self._validate_replay_assignment(
                assignments=self._replay_assignments, cursor=self._replay_cursor,
                input_y=input_y, input_x=input_x, sides=sides,
                expected_h=values.shape[-2], expected_w=values.shape[-1], expected_arity=2,
            )
        return self.reduce_tile_for_backward(payload, valid_mask, self.extra_state_for_backward()), gradient

    def init_reduction_state(self, *, batch_size, channels, device, dtype, accumulator_dtype):
        self.running_numerator = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=accumulator_dtype)
        self.running_denominator = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=accumulator_dtype)

    def accumulate_valid_tile(self, tile, valid_mask):
        values, logits = self._parse_multi_input_payload(tile)
        if self.running_numerator.numel() == 0:
            self.reset_stream_state(values.shape[0], values.shape[1], values.device, values.dtype)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, values.dtype)
        scores = torch.sigmoid(_normalize_logits(logits, values).to(dtype))
        valid = valid_mask[None, None].to(device=values.device, dtype=torch.bool)
        scores = torch.where(valid, scores, torch.zeros_like(scores))
        self.running_numerator = self.running_numerator.to(dtype) + (scores * values.to(dtype)).sum((-2, -1), keepdim=True, dtype=dtype)
        self.running_denominator = self.running_denominator.to(dtype) + scores.sum((-2, -1), keepdim=True, dtype=dtype)

    def finalize_from_state(self):
        if self.running_numerator.numel() == 0:
            raise RuntimeError("Streaming normalized-sigmoid attention state is empty.")
        denominator = self.running_denominator
        safe = denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
        return torch.where(denominator > 0, self.running_numerator / safe, torch.zeros_like(self.running_numerator)).to(self.running_sum.dtype)

    def extra_state_for_backward(self):
        denominator = self.running_denominator
        safe = denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
        mean = torch.where(denominator > 0, self.running_numerator / safe, torch.zeros_like(self.running_numerator))
        return {"denominator": safe.detach(), "mean": mean.detach(), "nonempty": (denominator > 0).detach()}

    def reduce_tile_for_backward(self, trimmed_output, valid_mask, global_context):
        values, logits = self._parse_multi_input_payload(trimmed_output)
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[-2:], device=values.device, dtype=torch.bool)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, values.dtype)
        scores = torch.sigmoid(_normalize_logits(logits, values).to(dtype))
        valid = valid_mask[None, None].to(device=values.device, dtype=torch.bool)
        scores = torch.where(valid, scores, torch.zeros_like(scores))
        contribution = scores * (values.to(dtype) - global_context["mean"].to(dtype))
        contribution = contribution / global_context["denominator"].to(dtype)
        contribution = torch.where(global_context["nonempty"].to(values.device), contribution, torch.zeros_like(contribution))
        return streaming_reduce_tile(contribution, valid_mask, None).to(values.dtype)

    def to_reducer(self):
        return NormalizedSigmoidAttentionReducer(
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
        )
