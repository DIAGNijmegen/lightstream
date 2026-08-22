"""Class-wise logit-attention pooling, offline and streaming."""

from __future__ import annotations

import torch

from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .sigmoid_attention import _TemperatureMixin
from .utils import prepare_spatial_mask, resolve_accumulator_dtype


class LogitAttentionPoolingReducer(_TemperatureMixin, BaseReducer):
    """Pool one ``[N,C,H,W]`` logit tensor with its spatial softmax."""

    def __init__(
        self,
        tau_init=1.0,
        learnable_temperature=False,
        stopgrad_attention=False,
        accumulator_dtype=None,
        mask_resize=False,
        mask_resize_mode="nearest",
        tau_min=1e-6,
    ):
        super().__init__()
        self._init_temperature(tau_init, tau_min, learnable_temperature)
        self.stopgrad_attention = bool(stopgrad_attention)
        self.accumulator_dtype = accumulator_dtype
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode

    def forward(self, *inputs, mask=None):
        if len(inputs) != 1:
            raise ValueError(
                f"LogitAttentionPoolingReducer expects exactly one input, got {len(inputs)}."
            )
        x = inputs[0]
        if x.ndim != 4:
            raise ValueError(
                f"Reducer expects one NCHW class-logit tensor, got shape={tuple(x.shape)}"
            )
        if self._streaming_passthrough:
            self._last_inputs = (x,)
            self._last_output = x
            return x

        dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        values = x.to(dtype)
        attention_logits = values.detach() if self.stopgrad_attention else values
        tau = self.current_tau.to(device=x.device, dtype=dtype)
        q = attention_logits / tau
        if mask is not None:
            valid = prepare_spatial_mask(
                mask,
                x,
                mask_resize=self.mask_resize,
                mask_resize_mode=self.mask_resize_mode,
            ).to(x.device)
            q = torch.where(valid, q, torch.full_like(q, torch.finfo(dtype).min))
        else:
            valid = None
        weights = torch.softmax(q.flatten(2), dim=-1).view_as(q)
        if valid is not None:
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            any_valid = valid.flatten(2).any(-1, keepdim=True).unsqueeze(-1)
        else:
            any_valid = torch.ones(
                (x.shape[0], 1, 1, 1), device=x.device, dtype=torch.bool
            )
        result = (weights * values).sum((-2, -1), keepdim=True, dtype=dtype)
        return torch.where(any_valid, result, torch.zeros_like(result)).to(x.dtype)

    def to_streaming(self):
        reducer = StreamingLogitAttentionPoolingReducer(
            tau_init=float(self.current_tau.detach()),
            learnable_temperature=self.learnable_temperature,
            stopgrad_attention=self.stopgrad_attention,
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
            tau_min=self.tau_min,
        )
        reducer.to(device=self.raw_tau.device, dtype=self.raw_tau.dtype)
        with torch.no_grad():
            reducer.raw_tau.copy_(self.raw_tau)
        return reducer


class StreamingLogitAttentionPoolingReducer(
    _TemperatureMixin, BaseStreamingGlobalReducer
):
    """Execution-only tiled implementation of logit-attention pooling."""

    def __init__(
        self,
        tau_init=1.0,
        learnable_temperature=False,
        stopgrad_attention=False,
        accumulator_dtype=None,
        mask_resize=False,
        mask_resize_mode="nearest",
        tau_min=1e-6,
    ):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
        self._init_temperature(tau_init, tau_min, learnable_temperature)
        self.stopgrad_attention = bool(stopgrad_attention)
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode
        for name in ("running_m", "running_zhat", "running_shat", "running_squarehat"):
            self.register_buffer(name, torch.zeros(0), persistent=False)
        self._temperature_surrogate_emitted = False

    def init_reduction_state(
        self, *, batch_size, channels, device, dtype, accumulator_dtype
    ):
        shape = (batch_size, channels, 1, 1)
        self.running_m = torch.full(
            shape,
            torch.finfo(accumulator_dtype).min,
            device=device,
            dtype=accumulator_dtype,
        )
        self.running_zhat = torch.zeros(shape, device=device, dtype=accumulator_dtype)
        self.running_shat = torch.zeros_like(self.running_zhat)
        self.running_squarehat = torch.zeros_like(self.running_zhat)

    def accumulate_valid_tile(self, tile, valid_mask):
        x = self._parse_single_input_payload(tile)
        if x.ndim != 4:
            raise ValueError(
                f"Reducer expects one NCHW class-logit tensor, got shape={tuple(x.shape)}"
            )
        if self.running_m.numel() == 0:
            self.reset_stream_state(x.shape[0], x.shape[1], x.device, x.dtype)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        values = x.to(dtype)
        attention_logits = values.detach() if self.stopgrad_attention else values
        q = attention_logits / self.current_tau.to(device=x.device, dtype=dtype)
        valid = valid_mask[None, None].to(x.device)
        q = torch.where(valid, q, torch.full_like(q, torch.finfo(dtype).min))
        tile_m = q.amax((-2, -1), keepdim=True)
        exp_shifted = torch.where(valid, torch.exp(q - tile_m), torch.zeros_like(q))
        tile_z = exp_shifted.sum((-2, -1), keepdim=True)
        tile_s = (exp_shifted * values).sum((-2, -1), keepdim=True)
        tile_square = (exp_shifted * values.square()).sum((-2, -1), keepdim=True)
        merged_m = torch.maximum(self.running_m.to(dtype), tile_m)
        old_scale = torch.exp(self.running_m.to(dtype) - merged_m)
        tile_scale = torch.exp(tile_m - merged_m)
        self.running_zhat = (
            self.running_zhat.to(dtype) * old_scale + tile_z * tile_scale
        )
        self.running_shat = (
            self.running_shat.to(dtype) * old_scale + tile_s * tile_scale
        )
        self.running_squarehat = (
            self.running_squarehat.to(dtype) * old_scale + tile_square * tile_scale
        )
        self.running_m = merged_m

    def finalize_from_state(self):
        if self.running_shat.numel() == 0:
            raise RuntimeError("Streaming logit-attention state is empty.")
        z = self.running_zhat.clamp_min(torch.finfo(self.running_zhat.dtype).tiny)
        result = torch.where(
            self.running_zhat > 0,
            self.running_shat / z,
            torch.zeros_like(self.running_shat),
        )
        return result.to(self.running_sum.dtype)

    def extra_state_for_backward(self):
        z = self.running_zhat.clamp_min(torch.finfo(self.running_zhat.dtype).tiny)
        mean = self.running_shat / z
        mean_square = self.running_squarehat / z
        tau = self.current_tau.to(device=z.device, dtype=z.dtype)
        return {
            "m": self.running_m.detach(),
            "zhat": z.detach(),
            "mean": mean.detach(),
            "dy_dtau": (-(mean_square - mean.square()) / tau.square()).detach(),
        }

    def start_backward_replay(self):
        super().start_backward_replay()
        self._temperature_surrogate_emitted = False

    def reduce_tile_for_backward(self, trimmed_output, valid_mask, global_context):
        x = self._parse_single_input_payload(trimmed_output)
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[-2:], device=x.device, dtype=torch.bool)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        values = x.to(dtype)
        attention_logits = values.detach() if self.stopgrad_attention else values
        tau = self.current_tau.to(device=x.device, dtype=dtype)
        q = attention_logits / tau.detach()
        valid = valid_mask[None, None].to(x.device)
        unnormalized = torch.where(
            valid,
            torch.exp(q - global_context["m"].to(q)),
            torch.zeros_like(q),
        )
        weights = unnormalized / global_context["zhat"].to(q)
        mean = global_context["mean"].to(q).detach()
        if self.stopgrad_attention:
            replay = weights.detach() * values
        else:
            replay = weights * (values - mean)
        reduced = streaming_reduce_tile(replay, valid_mask, None)
        if not self._temperature_surrogate_emitted:
            reduced = reduced + global_context["dy_dtau"].to(reduced) * (
                tau - tau.detach()
            )
            self._temperature_surrogate_emitted = True
        return reduced.to(x.dtype)

    def to_reducer(self):
        reducer = LogitAttentionPoolingReducer(
            tau_init=float(self.current_tau.detach()),
            learnable_temperature=self.learnable_temperature,
            stopgrad_attention=self.stopgrad_attention,
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
            tau_min=self.tau_min,
        )
        reducer.to(device=self.raw_tau.device, dtype=self.raw_tau.dtype)
        with torch.no_grad():
            reducer.raw_tau.copy_(self.raw_tau)
        return reducer
