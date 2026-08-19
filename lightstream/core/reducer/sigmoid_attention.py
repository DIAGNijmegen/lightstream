"""Class-wise sigmoid-attention pooling, offline and streaming."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import prepare_spatial_mask, resolve_accumulator_dtype


class _TemperatureMixin:
    def _init_temperature(self, tau_init, tau_min, learnable):
        tau, minimum = float(tau_init), float(tau_min)
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError("tau_min must be finite and non-negative.")
        if not math.isfinite(tau) or tau <= minimum:
            raise ValueError("tau_init must be finite and greater than tau_min.")
        raw = torch.tensor(math.log(math.expm1(tau - minimum)), dtype=torch.float32)
        self.tau_min, self.learnable_temperature = minimum, bool(learnable)
        if learnable:
            self.raw_tau = torch.nn.Parameter(raw)
        else:
            self.register_buffer("raw_tau", raw)

    @property
    def current_tau(self):
        return self.raw_tau.new_tensor(self.tau_min) + F.softplus(self.raw_tau)


class SigmoidAttentionPoolingReducer(_TemperatureMixin, BaseReducer):
    """Pool one ``[N,C,H,W]`` class-logit tensor to ``[N,C,1,1]``.

    The sole positional input supplies both the raw values being pooled and the
    logits from which attention is computed. ``mask`` is optional keyword-only
    spatial metadata; no separate value or attention tensor is required.
    """

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
                f"SigmoidAttentionPoolingReducer expects exactly one input, got {len(inputs)}."
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
        scores = torch.sigmoid(values)
        if self.stopgrad_attention:
            scores = scores.detach()
        q = scores / self.current_tau.to(device=x.device, dtype=dtype)
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
        m = q.amax(dim=(-2, -1), keepdim=True)
        exp_shifted = torch.exp(q - m)
        if valid is not None:
            exp_shifted = torch.where(valid, exp_shifted, torch.zeros_like(exp_shifted))
            any_valid = valid.flatten(2).any(-1, keepdim=True).unsqueeze(-1)
        else:
            any_valid = torch.ones(
                (x.shape[0], 1, 1, 1), device=x.device, dtype=torch.bool
            )
        z = exp_shifted.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        w = exp_shifted / z.clamp_min(torch.finfo(dtype).tiny)
        y = (w * values).sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        return torch.where(any_valid, y, torch.zeros_like(y)).to(x.dtype)

    def to_streaming(self):
        r = StreamingSigmoidAttentionPoolingReducer(
            tau_init=float(self.current_tau.detach()),
            learnable_temperature=self.learnable_temperature,
            stopgrad_attention=self.stopgrad_attention,
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
            tau_min=self.tau_min,
        )
        r.raw_tau.data.copy_(self.raw_tau.data.to(r.raw_tau))
        return r


class StreamingSigmoidAttentionPoolingReducer(
    _TemperatureMixin, BaseStreamingGlobalReducer
):
    """Execution-only tiled implementation of sigmoid-attention pooling."""

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
        for n in ("running_m", "running_zhat", "running_shat"):
            self.register_buffer(n, torch.zeros(0), persistent=False)

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
        scores = torch.sigmoid(values)
        if self.stopgrad_attention:
            scores = scores.detach()
        q = scores / self.current_tau.to(device=x.device, dtype=dtype)
        valid = valid_mask[None, None].to(x.device)
        q = torch.where(valid, q, torch.full_like(q, torch.finfo(dtype).min))
        mt = q.amax((-2, -1), keepdim=True)
        e = torch.where(valid, torch.exp(q - mt), torch.zeros_like(q))
        z = e.sum((-2, -1), keepdim=True)
        s = (e * values).sum((-2, -1), keepdim=True)
        mn = torch.maximum(self.running_m.to(dtype), mt)
        a = torch.exp(self.running_m.to(dtype) - mn)
        b = torch.exp(mt - mn)
        self.running_zhat = self.running_zhat.to(dtype) * a + z * b
        self.running_shat = self.running_shat.to(dtype) * a + s * b
        self.running_m = mn

    def finalize_from_state(self):
        if self.running_shat.numel() == 0:
            raise RuntimeError("Streaming sigmoid-attention state is empty.")
        z = self.running_zhat.clamp_min(torch.finfo(self.running_zhat.dtype).tiny)
        return torch.where(
            self.running_zhat > 0,
            self.running_shat / z,
            torch.zeros_like(self.running_shat),
        ).to(self.running_sum.dtype)

    def extra_state_for_backward(self):
        z = self.running_zhat.clamp_min(torch.finfo(self.running_zhat.dtype).tiny)
        return {
            "m": self.running_m.detach(),
            "zhat": z.detach(),
            "mean": (self.running_shat / z).detach(),
        }

    def reduce_tile_for_backward(self, trimmed_output, valid_mask, global_context):
        x = self._parse_single_input_payload(trimmed_output)
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[-2:], device=x.device, dtype=torch.bool)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        values = x.to(dtype)
        scores = torch.sigmoid(values)
        if self.stopgrad_attention:
            scores = scores.detach()
        q = scores / self.current_tau.to(device=x.device, dtype=dtype)
        valid = valid_mask[None, None].to(x.device)
        weights_unnorm = torch.where(
            valid, torch.exp(q - global_context["m"].to(q)), torch.zeros_like(q)
        )
        zhat = global_context["zhat"].to(q)
        mean = global_context["mean"].to(q).detach()

        # Represents the direct value derivative: d(sum_i w_i * value_i)/d(value_i)
        # with the attention weights held constant.
        direct_value_surrogate = streaming_reduce_tile(
            weights_unnorm.detach() * values, valid_mask, zhat
        )

        # Represents the normalized-attention derivative:
        # d(mean)/d(q_i) = w_i * (value_i - mean).  Detaching the centered
        # values keeps this path limited to sigmoid-score/temperature gradients.
        centered_attention_surrogate = streaming_reduce_tile(
            weights_unnorm * (values.detach() - mean), valid_mask, zhat
        )
        return (direct_value_surrogate + centered_attention_surrogate).to(x.dtype)

    def to_reducer(self):
        r = SigmoidAttentionPoolingReducer(
            tau_init=float(self.current_tau.detach()),
            learnable_temperature=self.learnable_temperature,
            stopgrad_attention=self.stopgrad_attention,
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
            tau_min=self.tau_min,
        )
        r.raw_tau.data.copy_(self.raw_tau.data.to(r.raw_tau))
        return r
