"""Fused-value multi-attention GeM reducers for offline and streaming execution."""

from __future__ import annotations

import torch

from .attention_gem import _normalize_logits, _validate_uniform_attention_eps
from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import prepare_spatial_mask, resolve_accumulator_dtype


_WEIGHT_LEN = 3


def _weights_tensor(name: str, weights: tuple[float, float, float]) -> torch.Tensor:
    if len(weights) != _WEIGHT_LEN:
        raise ValueError(f"{name} must contain exactly 3 weights, got {len(weights)}.")
    tensor = torch.tensor(tuple(float(weight) for weight in weights), dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _validate_value_maps(y1: torch.Tensor, y2: torch.Tensor, y3: torch.Tensor) -> None:
    for idx, y in enumerate((y1, y2, y3), start=1):
        if y.ndim != 4:
            raise ValueError(f"y{idx} must be an NCHW tensor, got shape={tuple(y.shape)}")
    if y2.shape != y1.shape or y3.shape != y1.shape:
        raise ValueError(
            "FusedAttentionGeMReducer value maps must have identical NCHW shapes; "
            f"got y1={tuple(y1.shape)}, y2={tuple(y2.shape)}, y3={tuple(y3.shape)}"
        )


def _normalize_three_logits(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    logits3: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _normalize_logits(logits1, reference),
        _normalize_logits(logits2, reference),
        _normalize_logits(logits3, reference),
    )


def _validate_stacked_logits(logits_stacked: torch.Tensor, reference: torch.Tensor) -> None:
    if logits_stacked.ndim != 4:
        raise ValueError(
            f"logits_stacked must be an NCHW tensor, got shape={tuple(logits_stacked.shape)}"
        )
    if (
        logits_stacked.shape[0] != reference.shape[0]
        or logits_stacked.shape[1] != 3
        or logits_stacked.shape[-2:] != reference.shape[-2:]
    ):
        raise ValueError(
            "logits_stacked must have shape [N,3,H,W] matching fused_y; "
            f"got logits_stacked={tuple(logits_stacked.shape)}, fused_y={tuple(reference.shape)}"
        )


class FusedAttentionGeMReducer(BaseReducer):
    """Fuse three value maps, then apply three globally normalized attention-GeM branches."""

    def __init__(
        self,
        r_init: float = 4.0,
        eps: float = 1e-6,
        value_weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
        attention_weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
        accumulator_dtype: torch.dtype | None = None,
        uniform_attention_eps: float = 0.0,
        mask_resize: bool = False,
        mask_resize_mode: str = "nearest",
    ):
        super().__init__()
        self.eps = float(eps)
        self.accumulator_dtype = accumulator_dtype
        self.uniform_attention_eps = _validate_uniform_attention_eps(uniform_attention_eps)
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode
        self.register_buffer("r", torch.tensor(float(r_init), dtype=torch.float32))
        self.register_buffer("value_weights", _weights_tensor("value_weights", value_weights))
        self.register_buffer("attention_weights", _weights_tensor("attention_weights", attention_weights))
        self._last_inputs = None
        self._last_output = None

    @property
    def current_r(self) -> torch.Tensor:
        return self.r

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if len(inputs) != 6:
            raise ValueError(
                "FusedAttentionGeMReducer expects exactly six inputs "
                f"(y1, y2, y3, logits1, logits2, logits3), got {len(inputs)}."
            )
        y1, y2, y3, logits1, logits2, logits3 = inputs
        _validate_value_maps(y1, y2, y3)
        logits = _normalize_three_logits(logits1, logits2, logits3, y1)

        if self._streaming_passthrough:
            vw = self.value_weights.to(device=y1.device, dtype=y1.dtype)
            fused_y = vw[0] * y1 + vw[1] * y2 + vw[2] * y3
            logits_stacked = torch.cat(logits, dim=1)
            passthrough = (fused_y.view_as(fused_y), logits_stacked.view_as(logits_stacked))
            self._last_inputs = passthrough
            self._last_output = passthrough[0]
            return passthrough

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, y1.dtype)
        vw = self.value_weights.to(device=y1.device, dtype=acc_dtype)
        aw = self.attention_weights.to(device=y1.device, dtype=acc_dtype)
        r = self.current_r.to(device=y1.device, dtype=acc_dtype)

        ys = (y1.to(dtype=acc_dtype), y2.to(device=y1.device, dtype=acc_dtype), y3.to(device=y1.device, dtype=acc_dtype))
        fused_y = vw[0] * ys[0] + vw[1] * ys[1] + vw[2] * ys[2]
        x_pow = fused_y.clamp_min(self.eps).pow(r)

        logits_acc = tuple(logit.to(device=y1.device, dtype=acc_dtype) for logit in logits)
        if mask is not None:
            mask_nchw = prepare_spatial_mask(mask, y1, mask_resize=self.mask_resize, mask_resize_mode=self.mask_resize_mode).to(device=y1.device)
            neg_inf = torch.finfo(acc_dtype).min
            logits_acc = tuple(torch.where(mask_nchw, logit, torch.full_like(logit, neg_inf)) for logit in logits_acc)
            any_valid = mask_nchw.flatten(2).any(dim=-1, keepdim=True).unsqueeze(-1)
        else:
            mask_nchw = None
            any_valid = torch.ones((y1.shape[0], 1, 1, 1), dtype=torch.bool, device=y1.device)

        branch_means = []
        for logit in logits_acc:
            m = logit.amax(dim=(-2, -1), keepdim=True)
            exp_shifted = torch.exp(logit - m)
            if mask_nchw is not None:
                exp_shifted = torch.where(mask_nchw, exp_shifted, torch.zeros_like(exp_shifted))
            z = exp_shifted.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
            weights = exp_shifted / z.clamp_min(self.eps)
            branch = (weights * x_pow).sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
            branch_means.append(torch.where(any_valid, branch, torch.zeros_like(branch)))

        attention_mean = aw[0] * branch_means[0] + aw[1] * branch_means[1] + aw[2] * branch_means[2]
        if self.uniform_attention_eps:
            if mask_nchw is not None:
                valid = mask_nchw.to(dtype=acc_dtype)
                uniform_z = valid.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).clamp_min(self.eps)
                uniform_mean = (x_pow * valid).sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype) / uniform_z
            else:
                uniform_mean = x_pow.mean(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
            mixed_mean = (1.0 - self.uniform_attention_eps) * attention_mean + self.uniform_attention_eps * uniform_mean
        else:
            mixed_mean = attention_mean

        return mixed_mean.clamp_min(self.eps).pow(1.0 / r).to(dtype=y1.dtype)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        reducer = StreamingFusedAttentionGeMReducer(
            r_init=float(self.current_r.detach().item()),
            eps=self.eps,
            value_weights=tuple(float(x) for x in self.value_weights.detach().cpu()),
            attention_weights=tuple(float(x) for x in self.attention_weights.detach().cpu()),
            accumulator_dtype=self.accumulator_dtype,
            uniform_attention_eps=self.uniform_attention_eps,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
        )
        reducer.r.data.copy_(self.current_r.detach().to(device=reducer.r.device, dtype=reducer.r.dtype))
        reducer.value_weights.data.copy_(self.value_weights.detach().to(device=reducer.value_weights.device, dtype=reducer.value_weights.dtype))
        reducer.attention_weights.data.copy_(self.attention_weights.detach().to(device=reducer.attention_weights.device, dtype=reducer.attention_weights.dtype))
        return reducer


class StreamingFusedAttentionGeMReducer(BaseStreamingGlobalReducer):
    """Streaming fused-value, three-attention GeM reducer."""

    def __init__(
        self,
        r_init: float = 4.0,
        eps: float = 1e-6,
        value_weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
        attention_weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
        accumulator_dtype: torch.dtype | None = None,
        uniform_attention_eps: float = 0.0,
        mask_resize: bool = False,
        mask_resize_mode: str = "nearest",
    ):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
        self.eps = float(eps)
        self.uniform_attention_eps = _validate_uniform_attention_eps(uniform_attention_eps)
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode
        self.register_buffer("r", torch.tensor(float(r_init), dtype=torch.float32))
        self.register_buffer("value_weights", _weights_tensor("value_weights", value_weights))
        self.register_buffer("attention_weights", _weights_tensor("attention_weights", attention_weights))
        self.register_buffer("running_m", torch.zeros(0), persistent=False)
        self.register_buffer("running_zhat", torch.zeros(0), persistent=False)
        self.register_buffer("running_shat", torch.zeros(0), persistent=False)
        self.register_buffer("running_valid_x_pow_sum", torch.zeros(0), persistent=False)
        self.register_buffer("running_valid_count", torch.zeros(0), persistent=False)
        # Backward-compatible aliases for the uniform-mixing path.
        self.register_buffer("running_uniform_sum", torch.zeros(0), persistent=False)
        self.register_buffer("running_uniform_count", torch.zeros(0), persistent=False)
        self._stream_output_dtype: torch.dtype | None = None

    @property
    def current_r(self) -> torch.Tensor:
        return self.r

    def forward(
        self,
        y1: torch.Tensor,
        y2: torch.Tensor,
        y3: torch.Tensor,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        logits3: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        _validate_value_maps(y1, y2, y3)
        vw = self.value_weights.to(device=y1.device, dtype=y1.dtype)
        fused_y = vw[0] * y1 + vw[1] * y2 + vw[2] * y3
        logits = _normalize_three_logits(logits1, logits2, logits3, y1)
        logits_stacked = torch.cat(logits, dim=1)
        passthrough = (fused_y.view_as(fused_y), logits_stacked.view_as(logits_stacked))
        self._last_inputs = passthrough
        self._last_output = passthrough[0]
        return passthrough

    def accumulate_stream_tile(self, trimmed_output, tile_y: int, tile_x: int, sides, dst_box, user_mask: torch.Tensor | None = None):
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 2:
            raise ValueError(f"StreamingFusedAttentionGeMReducer expects payload arity=2, got {len(payload)}")
        fused_y, logits_stacked = payload
        if fused_y.ndim != 4:
            raise ValueError(f"fused_y must be an NCHW tensor, got shape={tuple(fused_y.shape)}")
        _validate_stacked_logits(logits_stacked, fused_y)
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
                    int(fused_y.shape[-2]),
                    int(fused_y.shape[-1]),
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
        """Build a replay tile pair using the mask prepared by ``StreamingCNN``.

        ``valid_mask`` is already overlap-safe for the backward tile, so this
        reducer applies it locally without retaining separate mask replay state.
        """
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 2:
            raise ValueError(f"StreamingFusedAttentionGeMReducer expects payload arity=2, got {len(payload)}")
        fused_y, logits_stacked = payload
        if fused_y.ndim != 4:
            raise ValueError(f"fused_y must be an NCHW tensor, got shape={tuple(fused_y.shape)}")
        _validate_stacked_logits(logits_stacked, fused_y)
        expected_h, expected_w = int(fused_y.shape[-2]), int(fused_y.shape[-1])
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
        self._stream_output_dtype = dtype
        self.running_m = torch.full((batch_size, 3, 1, 1, 1), torch.finfo(accumulator_dtype).min, device=device, dtype=accumulator_dtype)
        self.running_zhat = torch.zeros((batch_size, 3, 1, 1, 1), device=device, dtype=accumulator_dtype)
        self.running_shat = torch.zeros((batch_size, 3, channels, 1, 1), device=device, dtype=accumulator_dtype)
        self.running_valid_x_pow_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=accumulator_dtype)
        self.running_valid_count = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=accumulator_dtype)
        self.running_uniform_sum = self.running_valid_x_pow_sum
        self.running_uniform_count = self.running_valid_count

    def accumulate_valid_tile(self, tile, valid_mask: torch.Tensor) -> None:
        payload = self._parse_multi_input_payload(tile)
        if len(payload) != 2:
            raise ValueError(f"StreamingFusedAttentionGeMReducer expects payload arity=2, got {len(payload)}")
        fused_y, logits_stacked = payload
        if fused_y.ndim != 4:
            raise ValueError(f"fused_y must be an NCHW tensor, got shape={tuple(fused_y.shape)}")
        _validate_stacked_logits(logits_stacked, fused_y)
        if self.running_shat.numel() == 0:
            self.reset_stream_state(batch_size=fused_y.shape[0], channels=fused_y.shape[1], device=fused_y.device, dtype=fused_y.dtype)

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, fused_y.dtype)
        r = self.current_r.to(device=fused_y.device, dtype=acc_dtype)
        x_pow = fused_y.to(acc_dtype).clamp_min(self.eps).pow(r)
        logits = logits_stacked.to(device=fused_y.device, dtype=acc_dtype)[:, :, None]

        neg_inf = torch.finfo(acc_dtype).min
        valid5d = valid_mask[None, None, None].to(device=fused_y.device, dtype=torch.bool)
        logits = torch.where(valid5d, logits, torch.full_like(logits, neg_inf))

        m_tile = logits.amax(dim=(-2, -1), keepdim=True)
        exp_tile = torch.exp(logits - m_tile)
        exp_tile = torch.where(valid5d, exp_tile, torch.zeros_like(exp_tile))
        z_tile = exp_tile.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
        s_tile = (exp_tile * x_pow[:, None]).sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)

        m_new = torch.maximum(self.running_m.to(dtype=acc_dtype), m_tile)
        alpha_prev = torch.exp(self.running_m.to(dtype=acc_dtype) - m_new)
        alpha_tile = torch.exp(m_tile - m_new)

        self.running_zhat = self.running_zhat.to(dtype=acc_dtype) * alpha_prev + z_tile * alpha_tile
        self.running_shat = self.running_shat.to(dtype=acc_dtype) * alpha_prev + s_tile * alpha_tile
        self.running_m = m_new
        valid = valid5d[:, 0].to(dtype=acc_dtype)
        valid_x_pow_sum = self.running_valid_x_pow_sum.to(dtype=acc_dtype) + (x_pow * valid).sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
        valid_count = self.running_valid_count.to(dtype=acc_dtype) + valid.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype)
        self.running_valid_x_pow_sum = valid_x_pow_sum
        self.running_valid_count = valid_count
        self.running_uniform_sum = valid_x_pow_sum
        self.running_uniform_count = valid_count

    def finalize_from_state(self) -> torch.Tensor:
        if self.running_shat.numel() == 0:
            raise RuntimeError("StreamingFusedAttentionGeMReducer state is empty, accumulate_stream_tile() was not called.")
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, self.running_shat.dtype)
        r = self.current_r.to(device=self.running_shat.device, dtype=acc_dtype)
        aw = self.attention_weights.to(device=self.running_shat.device, dtype=acc_dtype)
        branch_means = self.running_shat.to(dtype=acc_dtype) / self.running_zhat.to(dtype=acc_dtype).clamp_min(self.eps)
        # Sum_j attention_weights[j] * E_{softmax(att_j)}[fused_y ** r],
        # which is equivalent to GeM over fused_y weighted by
        # fused_a = sum_j attention_weights[j] * softmax(att_j).
        attention_mean = (branch_means * aw.view(1, 3, 1, 1, 1)).sum(dim=1)
        uniform_mean = self.running_valid_x_pow_sum.to(dtype=acc_dtype) / self.running_valid_count.to(dtype=acc_dtype).clamp_min(self.eps)
        mixed_mean = (1.0 - self.uniform_attention_eps) * attention_mean + self.uniform_attention_eps * uniform_mean
        output_dtype = self._stream_output_dtype or self.running_shat.dtype
        return mixed_mean.clamp_min(self.eps).pow(1.0 / r).to(dtype=output_dtype)

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, self.running_shat.dtype)
        zhat = self.running_zhat.to(dtype=acc_dtype).clamp_min(self.eps)
        attention_branch_means = self.running_shat.to(dtype=acc_dtype) / zhat
        aw = self.attention_weights.to(device=self.running_shat.device, dtype=acc_dtype)
        attention_mean = (attention_branch_means * aw.view(1, 3, 1, 1, 1)).sum(dim=1)
        valid_count = self.running_valid_count.to(dtype=acc_dtype).clamp_min(self.eps)
        uniform_mean = self.running_valid_x_pow_sum.to(dtype=acc_dtype) / valid_count
        final_mixed_mean = (1.0 - self.uniform_attention_eps) * attention_mean + self.uniform_attention_eps * uniform_mean
        r = self.current_r.to(device=self.running_shat.device, dtype=acc_dtype)
        return {
            # Existing per-branch softmax normalization state.
            "m": self.running_m.to(dtype=acc_dtype),
            "zhat": zhat,
            # Keep per-branch attention means separate from the fused pure-attention mean.
            "attention_branch_means": attention_branch_means,
            "branch_means": attention_branch_means,
            "attention_mean": attention_mean,
            "uniform_mean": uniform_mean,
            "final_mixed_mean": final_mixed_mean,
            "valid_count": valid_count,
            "uniform_attention_eps": float(self.uniform_attention_eps),
            "r": r,
            "attention_weights": aw,
            # Backward-compatible aliases for older callers/tests.
            "uniform_count": valid_count,
            "weighted_mean": final_mixed_mean,
        }

    def reduce_tile_for_backward(self, trimmed_output, valid_mask: torch.Tensor | None, global_context):
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 2:
            raise ValueError(f"StreamingFusedAttentionGeMReducer expects payload arity=2, got {len(payload)}")
        fused_y, logits_stacked = payload
        if valid_mask is None:
            valid_mask = torch.ones(fused_y.shape[-2:], device=fused_y.device, dtype=torch.bool)
        if fused_y.ndim != 4:
            raise ValueError(f"fused_y must be an NCHW tensor, got shape={tuple(fused_y.shape)}")
        _validate_stacked_logits(logits_stacked, fused_y)
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, fused_y.dtype)
        aw = global_context["attention_weights"].to(device=fused_y.device, dtype=acc_dtype)
        r = global_context["r"].to(device=fused_y.device, dtype=acc_dtype)
        m = global_context["m"].to(device=fused_y.device, dtype=acc_dtype)
        zhat = global_context["zhat"].to(device=fused_y.device, dtype=acc_dtype).clamp_min(self.eps)
        attention_branch_means_tensor = global_context["attention_branch_means"] if "attention_branch_means" in global_context else global_context["branch_means"]
        final_mixed_mean_tensor = global_context["final_mixed_mean"] if "final_mixed_mean" in global_context else global_context["weighted_mean"]
        attention_branch_means = attention_branch_means_tensor.to(device=fused_y.device, dtype=acc_dtype)
        final_mixed_mean = final_mixed_mean_tensor.to(device=fused_y.device, dtype=acc_dtype)
        uniform_attention_eps = float(global_context.get("uniform_attention_eps", self.uniform_attention_eps))

        x_pow = fused_y.to(acc_dtype).clamp_min(self.eps).pow(r)
        logits = logits_stacked.to(device=fused_y.device, dtype=acc_dtype)[:, :, None]

        valid5d = valid_mask[None, None, None].to(device=fused_y.device, dtype=torch.bool)
        neg_inf = torch.finfo(acc_dtype).min
        logits = torch.where(valid5d, logits, torch.full_like(logits, neg_inf))
        weights_unnorm = torch.exp(logits - m)
        weights_unnorm = torch.where(valid5d, weights_unnorm, torch.zeros_like(weights_unnorm))
        n, branches, _, h, w = weights_unnorm.shape
        channels = x_pow.shape[1]
        local_s_over_z = streaming_reduce_tile(
            (weights_unnorm * x_pow[:, None]).reshape(n * branches, channels, h, w),
            valid_mask,
            zhat.reshape(n * branches, 1, 1, 1),
        ).reshape(n, branches, channels, 1, 1)
        local_z_over_z = streaming_reduce_tile(
            weights_unnorm.reshape(n * branches, 1, h, w),
            valid_mask,
            zhat.reshape(n * branches, 1, 1, 1),
        ).reshape(n, branches, 1, 1, 1)

        branch_terms = local_s_over_z - attention_branch_means.detach() * local_z_over_z
        attention_replay_term = (branch_terms * aw.view(1, 3, 1, 1, 1)).sum(dim=1)
        replay_term = (1.0 - uniform_attention_eps) * attention_replay_term
        if uniform_attention_eps:
            valid_count_tensor = global_context["valid_count"] if "valid_count" in global_context else global_context["uniform_count"]
            valid_count = valid_count_tensor.to(device=fused_y.device, dtype=acc_dtype).clamp_min(self.eps)
            uniform_value_term = streaming_reduce_tile(x_pow, valid_mask, None) / valid_count
            replay_term = replay_term + uniform_attention_eps * uniform_value_term
        scale = (1.0 / r) * final_mixed_mean.clamp_min(self.eps).pow(1.0 / r - 1.0)
        surrogate = scale.detach() * replay_term
        return surrogate.to(dtype=fused_y.dtype)

    def to_reducer(self) -> FusedAttentionGeMReducer:
        reducer = FusedAttentionGeMReducer(
            r_init=float(self.current_r.detach().item()),
            eps=self.eps,
            value_weights=tuple(float(x) for x in self.value_weights.detach().cpu()),
            attention_weights=tuple(float(x) for x in self.attention_weights.detach().cpu()),
            accumulator_dtype=self.accumulator_dtype,
            uniform_attention_eps=self.uniform_attention_eps,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
        )
        reducer.r.data.copy_(self.current_r.detach().to(device=reducer.r.device, dtype=reducer.r.dtype))
        reducer.value_weights.data.copy_(self.value_weights.detach().to(device=reducer.value_weights.device, dtype=reducer.value_weights.dtype))
        reducer.attention_weights.data.copy_(self.attention_weights.detach().to(device=reducer.attention_weights.device, dtype=reducer.attention_weights.dtype))
        return reducer