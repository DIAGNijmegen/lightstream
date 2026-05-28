"""Attention-weighted GeM reducers for offline and streaming execution."""

from __future__ import annotations

import torch

from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import normalize_spatial_mask, resolve_accumulator_dtype


def _normalize_logits(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Normalize attention logits to ``[N, 1, H, W]`` for broadcasting."""
    if logits.ndim == 3:
        if logits.shape[0] != x.shape[0] or logits.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"3D logits shape {tuple(logits.shape)} must be [N,H,W] with N={x.shape[0]}, H/W={tuple(x.shape[-2:])}"
            )
        return logits[:, None].to(device=x.device)

    if logits.ndim == 4:
        if logits.shape[0] != x.shape[0] or logits.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"4D logits shape {tuple(logits.shape)} must be [N,1|C,H,W] with N={x.shape[0]}, H/W={tuple(x.shape[-2:])}"
            )
        if logits.shape[1] not in (1, x.shape[1]):
            raise ValueError(f"4D logits channel dim must be 1 or C={x.shape[1]}, got {logits.shape[1]}")
        if logits.shape[1] == x.shape[1]:
            logits = logits.mean(dim=1, keepdim=True)
        return logits.to(device=x.device)

    raise ValueError(f"logits must be 3D/4D spatial tensor, got shape={tuple(logits.shape)}")


class AttentionGeMReducer(BaseReducer):
    """Apply attention-weighted global GeM reduction on NCHW tensors."""

    def __init__(self, r_init: float = 4.0, eps: float = 1e-6, accumulator_dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = float(eps)
        self.accumulator_dtype = accumulator_dtype
        self.register_buffer("r", torch.tensor(float(r_init), dtype=torch.float32))

    @property
    def current_r(self) -> torch.Tensor:
        return self.r

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(inputs) != 2:
            raise ValueError(f"AttentionGeMReducer expects exactly two inputs (x, attn_logits), got {len(inputs)}.")
        x, attn_logits = inputs
        if x.ndim != 4:
            raise ValueError(f"Reducer expects NCHW x tensor, got shape={tuple(x.shape)}")
        logits = _normalize_logits(attn_logits, x)
        if self._streaming_passthrough:
            logits_term = logits.to(dtype=x.dtype).sum(dim=(-2, -1), keepdim=True)
            graph_passthrough = logits_term - logits_term.detach()
            x_passthrough = x + graph_passthrough
            return x_passthrough, logits

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        x_acc = x.to(dtype=acc_dtype)
        x_pow = x_acc.clamp_min(self.eps).pow(self.current_r.to(device=x.device, dtype=acc_dtype))
        logits_acc = logits.to(dtype=acc_dtype)

        if mask is not None:
            mask_nchw = normalize_spatial_mask(mask, x).to(device=x.device)
            neg_inf = torch.finfo(acc_dtype).min
            logits_acc = torch.where(mask_nchw, logits_acc, torch.full_like(logits_acc, neg_inf))
            any_valid = mask_nchw.flatten(2).any(dim=-1, keepdim=True).unsqueeze(-1)
        else:
            any_valid = torch.ones((x.shape[0], 1, 1, 1), dtype=torch.bool, device=x.device)

        m = logits_acc.amax(dim=(-2, -1), keepdim=True)
        exp_shifted = torch.exp(logits_acc - m)
        if mask is not None:
            exp_shifted = torch.where(mask_nchw, exp_shifted, torch.zeros_like(exp_shifted))

        z = exp_shifted.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
        weights = exp_shifted / z.clamp_min(self.eps)
        weighted = (weights * x_pow).sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)

        # If a sample is fully masked, return a numerically safe zero contribution.
        weighted = torch.where(any_valid, weighted, torch.zeros_like(weighted))
        y = weighted.clamp_min(self.eps).pow(1.0 / self.current_r.to(device=x.device, dtype=acc_dtype))
        return y.to(dtype=x.dtype)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        reducer = StreamingAttentionGeMReducer(
            r_init=float(self.current_r.detach().item()),
            eps=self.eps,
            accumulator_dtype=self.accumulator_dtype,
        )
        reducer.r.data.copy_(self.current_r.detach().to(device=reducer.r.device, dtype=reducer.r.dtype))
        return reducer


class StreamingAttentionGeMReducer(BaseStreamingGlobalReducer):
    """Streaming attention-weighted global GeM reducer with stable softmax accumulation."""

    def __init__(self, r_init: float = 4.0, eps: float = 1e-6, accumulator_dtype: torch.dtype | None = None):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
        self.eps = float(eps)
        self.register_buffer("r", torch.tensor(float(r_init), dtype=torch.float32))
        self.register_buffer("running_m", torch.zeros(0), persistent=False)
        self.register_buffer("running_zhat", torch.zeros(0), persistent=False)
        self.register_buffer("running_shat", torch.zeros(0), persistent=False)

    @property
    def current_r(self) -> torch.Tensor:
        return self.r

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(inputs) != 2:
            raise ValueError(f"StreamingAttentionGeMReducer expects exactly two inputs (x_tile, logits_tile), got {len(inputs)}.")
        x_tile, logits_tile = inputs
        self._last_inputs = (x_tile, logits_tile)
        self._last_output = x_tile
        return x_tile, logits_tile

    def accumulate_stream_tile(self, trimmed_output, tile_y: int, tile_x: int, sides, dst_box, user_mask: torch.Tensor | None = None):
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 2:
            raise ValueError(f"StreamingAttentionGeMReducer expects payload arity=2, got {len(payload)}")
        x_tile, _ = payload
        dst_y0, dst_y1, dst_x0, dst_x1 = dst_box
        seen_slice = self._stream_seen_mask[dst_y0:dst_y1, dst_x0:dst_x1]
        new_mask = ~seen_slice
        effective_mask = new_mask if user_mask is None else (new_mask & user_mask.to(dtype=torch.bool, device=new_mask.device))
        if self._debug_replay_enabled:
            if self._replay_assignments is None:
                raise RuntimeError("Reducer replay assignments are not initialized.")
            self._replay_assignments.append(
                (
                    int(tile_y),
                    int(tile_x),
                    bool(sides.top),
                    bool(sides.left),
                    bool(sides.right),
                    bool(sides.bottom),
                    int(x_tile.shape[-2]),
                    int(x_tile.shape[-1]),
                    int(dst_y0),
                    int(dst_y1),
                    int(dst_x0),
                    int(dst_x1),
                    len(payload),
                )
            )
        if torch.any(effective_mask):
            self.accumulate_valid_tile(payload, valid_mask=effective_mask)
        seen_slice |= new_mask

    def build_backward_pair(self, trimmed_output, gradient: torch.Tensor, *, input_y: int, input_x: int, sides, valid_mask: torch.Tensor | None = None):
        payload = self._parse_multi_input_payload(trimmed_output)
        x_tile = payload[0]
        expected_h, expected_w = int(x_tile.shape[-2]), int(x_tile.shape[-1])
        if self._debug_replay_enabled:
            if self._replay_assignments is None or self._replay_cursor is None:
                raise RuntimeError("Reducer replay state is not initialized. Call start_backward_replay() first.")
            self._replay_cursor = self._validate_replay_assignment(
                assignments=self._replay_assignments,
                cursor=self._replay_cursor,
                input_y=input_y,
                input_x=input_x,
                sides=sides,
                expected_h=expected_h,
                expected_w=expected_w,
                expected_arity=len(payload),
            )
        reduced_output = self.reduce_tile_for_backward(payload, valid_mask=valid_mask, global_context=self.extra_state_for_backward())
        return reduced_output, gradient

    def init_reduction_state(self, *, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, accumulator_dtype: torch.dtype) -> None:
        self.running_m = torch.full((batch_size, 1, 1, 1), torch.finfo(accumulator_dtype).min, device=device, dtype=accumulator_dtype)
        self.running_zhat = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=accumulator_dtype)
        self.running_shat = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=accumulator_dtype)

    def accumulate_valid_tile(self, tile, valid_mask: torch.Tensor) -> None:
        x_tile, logits_tile = self._parse_multi_input_payload(tile)
        if self.running_shat.numel() == 0:
            self.reset_stream_state(batch_size=x_tile.shape[0], channels=x_tile.shape[1], device=x_tile.device, dtype=x_tile.dtype)

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x_tile.dtype)
        x = x_tile.to(dtype=acc_dtype).clamp_min(self.eps)
        logits = _normalize_logits(logits_tile, x_tile).to(dtype=acc_dtype)
        r = self.current_r.to(device=x_tile.device, dtype=acc_dtype)
        x_pow = x.pow(r)

        neg_inf = torch.finfo(acc_dtype).min
        valid4d = valid_mask[None, None].to(device=x_tile.device, dtype=torch.bool)
        logits = torch.where(valid4d, logits, torch.full_like(logits, neg_inf))

        m_tile = logits.amax(dim=(-2, -1), keepdim=True)
        exp_tile = torch.exp(logits - m_tile)
        exp_tile = torch.where(valid4d, exp_tile, torch.zeros_like(exp_tile))
        z_tile = exp_tile.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
        s_tile = (exp_tile * x_pow).sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)

        m_new = torch.maximum(self.running_m.to(dtype=acc_dtype), m_tile)
        alpha_prev = torch.exp(self.running_m.to(dtype=acc_dtype) - m_new)
        alpha_tile = torch.exp(m_tile - m_new)

        self.running_zhat = self.running_zhat.to(dtype=acc_dtype) * alpha_prev + z_tile * alpha_tile
        self.running_shat = self.running_shat.to(dtype=acc_dtype) * alpha_prev + s_tile * alpha_tile
        self.running_m = m_new

    def finalize_from_state(self) -> torch.Tensor:
        if self.running_shat.numel() == 0:
            raise RuntimeError("StreamingAttentionGeMReducer state is empty, accumulate_stream_tile() was not called.")
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, self.running_shat.dtype)
        r = self.current_r.to(device=self.running_shat.device, dtype=acc_dtype)
        weighted_mean = self.running_shat.to(dtype=acc_dtype) / self.running_zhat.to(dtype=acc_dtype).clamp_min(self.eps)
        return weighted_mean.clamp_min(self.eps).pow(1.0 / r).to(dtype=self.running_shat.dtype)

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, self.running_shat.dtype)
        return {
            "m": self.running_m.to(dtype=acc_dtype),
            "zhat": self.running_zhat.to(dtype=acc_dtype).clamp_min(self.eps),
            "r": self.current_r.to(device=self.running_shat.device, dtype=acc_dtype),
        }

    def reduce_tile_for_backward(self, trimmed_output, valid_mask: torch.Tensor | None, global_context):
        if valid_mask is None:
            raise ValueError("StreamingAttentionGeMReducer backward replay requires a valid_mask.")
        x_tile, logits_tile = self._parse_multi_input_payload(trimmed_output)
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, x_tile.dtype)
        x = x_tile.to(dtype=acc_dtype).clamp_min(self.eps)
        logits = _normalize_logits(logits_tile, x_tile).to(dtype=acc_dtype)
        r = global_context["r"].to(device=x_tile.device, dtype=acc_dtype)
        m = global_context["m"].to(device=x_tile.device, dtype=acc_dtype)
        zhat = global_context["zhat"].to(device=x_tile.device, dtype=acc_dtype).clamp_min(self.eps)

        valid4d = valid_mask[None, None].to(device=x_tile.device, dtype=torch.bool)
        weights_unnorm = torch.exp(logits - m)
        weights_unnorm = torch.where(valid4d, weights_unnorm, torch.zeros_like(weights_unnorm))
        weighted = streaming_reduce_tile(weights_unnorm * x.pow(r), valid_mask, zhat)
        return weighted.clamp_min(self.eps).pow(1.0 / r).to(dtype=x_tile.dtype)

    def to_reducer(self) -> AttentionGeMReducer:
        reducer = AttentionGeMReducer(
            r_init=float(self.current_r.detach().item()),
            eps=self.eps,
            accumulator_dtype=self.accumulator_dtype,
        )
        reducer.r.data.copy_(self.current_r.detach().to(device=reducer.r.device, dtype=reducer.r.dtype))
        return reducer
