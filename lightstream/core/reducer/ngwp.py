"""Normalized global weighted pooling reducers."""

from __future__ import annotations

import math

import torch

from .base import BaseStreamingGlobalReducer
from .reducer_base import BaseReducer
from .utils import prepare_spatial_mask, resolve_accumulator_dtype


def _validate_inputs(scores: torch.Tensor, activation_masks: torch.Tensor) -> None:
    if not isinstance(scores, torch.Tensor) or not isinstance(activation_masks, torch.Tensor):
        raise TypeError("NGWPReducer scores and activation_masks must both be tensors.")
    if scores.ndim != 4 or activation_masks.ndim != 4:
        raise ValueError("NGWPReducer inputs must both be NCHW [N, C, H, W] tensors.")
    if scores.shape != activation_masks.shape:
        raise ValueError(
            "NGWPReducer scores and activation_masks must have identical NCHW shapes; "
            f"got {tuple(scores.shape)} and {tuple(activation_masks.shape)}."
        )


class NGWPReducer(BaseReducer):
    """Reduce ``(scores, activation_masks)`` by normalized global weighting."""

    def __init__(self, eps: float = 1, accumulator_dtype: torch.dtype | None = None,
                 mask_resize: bool = False, mask_resize_mode: str = "nearest"):
        super().__init__()
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("eps must be finite and non-negative.")
        self.eps = float(eps)
        self.accumulator_dtype = accumulator_dtype
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(inputs) != 2:
            raise ValueError(f"NGWPReducer expects (scores, activation_masks), got {len(inputs)} inputs.")
        scores, activation_masks = inputs
        _validate_inputs(scores, activation_masks)
        if self._streaming_passthrough:
            passthrough = (
                scores.view_as(scores),
                activation_masks.view_as(activation_masks),
            )
            self._last_inputs = passthrough
            self._last_output = passthrough[0]
            return passthrough
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, scores.dtype)
        scores_acc, activations_acc = scores.to(acc_dtype), activation_masks.to(acc_dtype)
        if mask is not None:
            tissue = prepare_spatial_mask(mask, scores, mask_resize=self.mask_resize, mask_resize_mode=self.mask_resize_mode).to(acc_dtype)
            scores_acc, activations_acc = scores_acc * tissue, activations_acc * tissue
        numerator = (activations_acc * scores_acc).sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
        denominator = activations_acc.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
        return (numerator / (denominator + self.eps)).to(dtype=scores.dtype)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        return StreamingNGWPReducer(self.eps, self.accumulator_dtype, self.mask_resize, self.mask_resize_mode)


class StreamingNGWPReducer(BaseStreamingGlobalReducer):
    """Streaming nGWP, accumulating global numerator and denominator separately."""

    def __init__(self, eps: float = 1, accumulator_dtype: torch.dtype | None = None,
                 mask_resize: bool = False, mask_resize_mode: str = "nearest"):
        super().__init__(mode="sum", accumulator_dtype=accumulator_dtype)
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("eps must be finite and non-negative.")
        self.eps, self.mask_resize, self.mask_resize_mode = float(eps), bool(mask_resize), mask_resize_mode
        self.register_buffer("running_activation_sum", torch.zeros(0), persistent=False)

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None):
        if len(inputs) != 2:
            raise ValueError(f"StreamingNGWPReducer expects (scores_tile, activation_masks_tile), got {len(inputs)} inputs.")
        scores_tile, activation_masks_tile = inputs
        _validate_inputs(scores_tile, activation_masks_tile)
        passthrough = (
            scores_tile.view_as(scores_tile),
            activation_masks_tile.view_as(activation_masks_tile),
        )
        self._last_inputs = passthrough
        self._last_output = passthrough[0]
        return passthrough

    def _payload(self, payload):
        values = self._parse_multi_input_payload(payload)
        if len(values) != 2:
            raise ValueError(f"StreamingNGWPReducer expects payload arity=2, got {len(values)}.")
        _validate_inputs(*values)
        return values

    def accumulate_stream_tile(self, trimmed_output, tile_y, tile_x, sides, dst_box, user_mask=None):
        payload = self._payload(trimmed_output)
        return self._accumulate_payload(payload, tile_y, tile_x, sides, dst_box, user_mask)

    def _accumulate_payload(self, payload, tile_y, tile_x, sides, dst_box, user_mask):
        scores, _ = payload
        y0, y1, x0, x1 = dst_box
        seen = self._stream_seen_mask[y0:y1, x0:x1]
        new, effective = ~seen, ~seen if user_mask is None else (~seen & user_mask.to(device=seen.device, dtype=torch.bool))
        if self._debug_replay_enabled:
            self._replay_assignments.append((int(tile_y), int(tile_x), bool(sides.top), bool(sides.left), bool(sides.right), bool(sides.bottom), scores.shape[-2], scores.shape[-1], y0, y1, x0, x1, 2))
        if torch.any(effective): self.accumulate_valid_tile(payload, effective)
        seen |= new

    def build_backward_pair(self, trimmed_output, gradient, *, input_y, input_x, sides, valid_mask=None):
        payload = self._payload(trimmed_output)
        if self._debug_replay_enabled:
            self._replay_cursor = self._validate_replay_assignment(assignments=self._replay_assignments, cursor=self._replay_cursor, input_y=input_y, input_x=input_x, sides=sides, expected_h=payload[0].shape[-2], expected_w=payload[0].shape[-1], expected_arity=2)
        return self.reduce_tile_for_backward(payload, valid_mask, self.extra_state_for_backward()), gradient

    def init_reduction_state(self, *, batch_size, channels, device, dtype, accumulator_dtype):
        self._output_dtype = dtype
        self.running_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=accumulator_dtype)
        self.running_activation_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=accumulator_dtype)

    def accumulate_valid_tile(self, tile, valid_mask):
        scores, activations = self._payload(tile)
        if self.running_sum.numel() == 0: self.reset_stream_state(scores.shape[0], scores.shape[1], scores.device, scores.dtype)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, scores.dtype)
        valid = valid_mask[None, None].to(device=scores.device, dtype=dtype)
        self.running_sum = self.running_sum + (scores.to(dtype) * activations.to(dtype) * valid).sum((-2, -1), keepdim=True, dtype=dtype)
        self.running_activation_sum = self.running_activation_sum + (activations.to(dtype) * valid).sum((-2, -1), keepdim=True, dtype=dtype)

    def finalize_from_state(self):
        if self.running_sum.numel() == 0: raise RuntimeError("StreamingNGWPReducer state is empty, accumulate_stream_tile() was not called.")
        return (self.running_sum / (self.running_activation_sum.to(self.running_sum.dtype) + self.eps)).to(self._output_dtype)

    def extra_state_for_backward(self):
        return {"weighted_sum": self.running_sum, "denominator": self.running_activation_sum + self.eps}

    def reduce_tile_for_backward(self, trimmed_output, valid_mask, global_context):
        scores, activations = self._payload(trimmed_output)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, scores.dtype)
        valid = 1 if valid_mask is None else valid_mask[None, None].to(device=scores.device, dtype=dtype)
        weighted = (scores.to(dtype) * activations.to(dtype) * valid).sum((-2, -1), keepdim=True, dtype=dtype)
        activation_sum = (activations.to(dtype) * valid).sum((-2, -1), keepdim=True, dtype=dtype)
        denom = global_context["denominator"].to(device=scores.device, dtype=dtype)
        total = global_context["weighted_sum"].to(device=scores.device, dtype=dtype)
        return (weighted / denom - activation_sum * total / denom.square()).to(scores.dtype)

    def to_reducer(self):
        return NGWPReducer(self.eps, self.accumulator_dtype, self.mask_resize, self.mask_resize_mode)
