"""Fused-value multi-attention GeM reducers for offline and streaming execution."""

from __future__ import annotations

import torch

from .attention_gem import _normalize_logits
from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import normalize_spatial_mask, resolve_accumulator_dtype


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


class FusedAttentionGeMReducer(BaseReducer):
    """Fuse three value maps, then apply three globally normalized attention-GeM branches."""

    def __init__(
        self,
        r_init: float = 4.0,
        eps: float = 1e-6,
        value_weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
        attention_weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
        accumulator_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = float(eps)
        self.accumulator_dtype = accumulator_dtype
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
            passthrough = tuple(t.view_as(t) for t in (y1, y2, y3, *logits))
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
            mask_nchw = normalize_spatial_mask(mask, y1).to(device=y1.device)
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

        weighted_mean = aw[0] * branch_means[0] + aw[1] * branch_means[1] + aw[2] * branch_means[2]
        return weighted_mean.clamp_min(self.eps).pow(1.0 / r).to(dtype=y1.dtype)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        reducer = StreamingFusedAttentionGeMReducer(
            r_init=float(self.current_r.detach().item()),
            eps=self.eps,
            value_weights=tuple(float(x) for x in self.value_weights.detach().cpu()),
            attention_weights=tuple(float(x) for x in self.attention_weights.detach().cpu()),
            accumulator_dtype=self.accumulator_dtype,
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
    ):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
        self.eps = float(eps)
        self.register_buffer("r", torch.tensor(float(r_init), dtype=torch.float32))
        self.register_buffer("value_weights", _weights_tensor("value_weights", value_weights))
        self.register_buffer("attention_weights", _weights_tensor("attention_weights", attention_weights))
        self.register_buffer("running_m", torch.zeros(0), persistent=False)
        self.register_buffer("running_zhat", torch.zeros(0), persistent=False)
        self.register_buffer("running_shat", torch.zeros(0), persistent=False)
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
        logits = _normalize_three_logits(logits1, logits2, logits3, y1)
        passthrough = tuple(t.view_as(t) for t in (y1, y2, y3, *logits))
        self._last_inputs = passthrough
        self._last_output = passthrough[0]
        return passthrough

    @staticmethod
    def _payload_spatial_shapes(payload: tuple[torch.Tensor, ...]) -> tuple[tuple[int, int], ...]:
        return tuple((int(t.shape[-2]), int(t.shape[-1])) for t in payload)

    def accumulate_stream_tile(self, trimmed_output, tile_y: int, tile_x: int, sides, dst_box, user_mask: torch.Tensor | None = None):
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 6:
            raise ValueError(f"StreamingFusedAttentionGeMReducer expects payload arity=6, got {len(payload)}")
        y1 = payload[0]
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
                    int(y1.shape[-2]),
                    int(y1.shape[-1]),
                    int(dst_y0),
                    int(dst_y1),
                    int(dst_x0),
                    int(dst_x1),
                    len(payload),
                    self._payload_spatial_shapes(payload),
                    int(effective_mask.sum().item()),
                    bool(user_mask is not None),
                )
            )
        if torch.any(effective_mask):
            self.accumulate_valid_tile(payload, valid_mask=effective_mask)
        seen_slice |= new_mask

    def build_backward_pair(self, trimmed_output, gradient: torch.Tensor, *, input_y: int, input_x: int, sides, valid_mask: torch.Tensor | None = None):
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 6:
            raise ValueError(f"StreamingFusedAttentionGeMReducer expects payload arity=6, got {len(payload)}")
        y1 = payload[0]
        expected_h, expected_w = int(y1.shape[-2]), int(y1.shape[-1])
        if self._debug_replay_enabled:
            if self._replay_assignments is None or self._replay_cursor is None:
                raise RuntimeError("Reducer replay state is not initialized. Call start_backward_replay() first.")
            backward_valid_pixels = None if valid_mask is None else int(valid_mask.sum().item())
            self._replay_cursor = self._validate_fused_replay_assignment(
                assignments=self._replay_assignments,
                cursor=self._replay_cursor,
                input_y=input_y,
                input_x=input_x,
                sides=sides,
                expected_h=expected_h,
                expected_w=expected_w,
                expected_arity=len(payload),
                backward_payload_shapes=self._payload_spatial_shapes(payload),
                backward_valid_pixels=backward_valid_pixels,
                backward_valid_mask_shape=(
                    None if valid_mask is None else (int(valid_mask.shape[-2]), int(valid_mask.shape[-1]))
                ),
            )
        reduced_output = self.reduce_tile_for_backward(payload, valid_mask=valid_mask, global_context=self.extra_state_for_backward())
        return reduced_output, gradient

    def _validate_fused_replay_assignment(
        self,
        *,
        assignments: list[tuple],
        cursor: int,
        input_y: int,
        input_x: int,
        sides,
        expected_h: int,
        expected_w: int,
        expected_arity: int,
        backward_payload_shapes: tuple[tuple[int, int], ...],
        backward_valid_pixels: int | None,
        backward_valid_mask_shape: tuple[int, int] | None,
    ) -> int:
        if cursor >= len(assignments):
            raise RuntimeError("Reducer assignment cursor out of range.")
        (
            f_tile_y,
            f_tile_x,
            f_top,
            f_left,
            f_right,
            f_bottom,
            f_h,
            f_w,
            dst_y0,
            dst_y1,
            dst_x0,
            dst_x1,
            f_arity,
            forward_payload_shapes,
            forward_effective_pixels,
            forward_user_mask_present,
        ) = assignments[cursor]
        if (
            int(input_y) != int(f_tile_y)
            or int(input_x) != int(f_tile_x)
            or bool(sides.top) != bool(f_top)
            or bool(sides.left) != bool(f_left)
            or bool(sides.right) != bool(f_right)
            or bool(sides.bottom) != bool(f_bottom)
        ):
            raise RuntimeError(
                "Reducer tile replay mismatch: "
                f"forward tile=({f_tile_y},{f_tile_x},{f_top},{f_left},{f_right},{f_bottom}) "
                f"backward tile=({int(input_y)},{int(input_x)},{bool(sides.top)},{bool(sides.left)},{bool(sides.right)},{bool(sides.bottom)})"
            )
        dst_box = (int(dst_y0), int(dst_y1), int(dst_x0), int(dst_x1))
        if expected_h != int(f_h) or expected_w != int(f_w):
            raise RuntimeError(
                "Reducer trimmed shape mismatch: "
                f"forward=({f_h},{f_w}) backward=({expected_h},{expected_w}) dst_box={dst_box}"
            )
        if (dst_y1 - dst_y0) != expected_h or (dst_x1 - dst_x0) != expected_w:
            raise RuntimeError(
                "Reducer assignment mismatch: "
                f"stored=({dst_y0}:{dst_y1},{dst_x0}:{dst_x1}) current=({expected_h},{expected_w})"
            )
        if int(f_arity) != int(expected_arity):
            raise RuntimeError(
                f"Reducer input arity mismatch: forward={int(f_arity)} "
                f"backward={int(expected_arity)} dst_box={dst_box}"
            )
        if tuple(forward_payload_shapes) != tuple(backward_payload_shapes):
            raise RuntimeError(
                "FusedAttentionGeM reducer payload spatial shape replay mismatch: "
                f"forward={tuple(forward_payload_shapes)} backward={tuple(backward_payload_shapes)} dst_box={dst_box}"
            )
        expected_mask_shape = (int(dst_y1 - dst_y0), int(dst_x1 - dst_x0))
        if backward_valid_mask_shape != expected_mask_shape:
            raise RuntimeError(
                "FusedAttentionGeM reducer backward valid mask shape mismatch: "
                f"forward dst_box={dst_box} expected_shape={expected_mask_shape} backward_shape={backward_valid_mask_shape}"
            )
        if backward_valid_pixels is None:
            raise RuntimeError("FusedAttentionGeM reducer backward replay requires valid_mask diagnostics.")
        if int(forward_effective_pixels) != int(backward_valid_pixels):
            raise RuntimeError(
                "FusedAttentionGeM reducer forward/backward effective pixel count mismatch: "
                f"forward_effective_pixels={int(forward_effective_pixels)} "
                f"backward_valid_pixels={int(backward_valid_pixels)} "
                f"dst_box={dst_box} payload_shapes={tuple(forward_payload_shapes)} "
                f"user_mask_present={bool(forward_user_mask_present)}"
            )
        return cursor + 1

    def init_reduction_state(self, *, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, accumulator_dtype: torch.dtype) -> None:
        self._stream_output_dtype = dtype
        self.running_m = torch.full((batch_size, 3, 1, 1, 1), torch.finfo(accumulator_dtype).min, device=device, dtype=accumulator_dtype)
        self.running_zhat = torch.zeros((batch_size, 3, 1, 1, 1), device=device, dtype=accumulator_dtype)
        self.running_shat = torch.zeros((batch_size, 3, channels, 1, 1), device=device, dtype=accumulator_dtype)

    def accumulate_valid_tile(self, tile, valid_mask: torch.Tensor) -> None:
        payload = self._parse_multi_input_payload(tile)
        if len(payload) != 6:
            raise ValueError(f"StreamingFusedAttentionGeMReducer expects payload arity=6, got {len(payload)}")
        y1, y2, y3, logits1, logits2, logits3 = payload
        _validate_value_maps(y1, y2, y3)
        if self.running_shat.numel() == 0:
            self.reset_stream_state(batch_size=y1.shape[0], channels=y1.shape[1], device=y1.device, dtype=y1.dtype)

        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, y1.dtype)
        vw = self.value_weights.to(device=y1.device, dtype=acc_dtype)
        r = self.current_r.to(device=y1.device, dtype=acc_dtype)
        fused_y = vw[0] * y1.to(dtype=acc_dtype) + vw[1] * y2.to(device=y1.device, dtype=acc_dtype) + vw[2] * y3.to(device=y1.device, dtype=acc_dtype)
        x_pow = fused_y.clamp_min(self.eps).pow(r)
        logits = torch.stack(
            [logit.to(device=y1.device, dtype=acc_dtype) for logit in _normalize_three_logits(logits1, logits2, logits3, y1)],
            dim=1,
        )

        neg_inf = torch.finfo(acc_dtype).min
        valid5d = valid_mask[None, None, None].to(device=y1.device, dtype=torch.bool)
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
        weighted_mean = (branch_means * aw.view(1, 3, 1, 1, 1)).sum(dim=1)
        output_dtype = self._stream_output_dtype or self.running_shat.dtype
        return weighted_mean.clamp_min(self.eps).pow(1.0 / r).to(dtype=output_dtype)

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, self.running_shat.dtype)
        zhat = self.running_zhat.to(dtype=acc_dtype).clamp_min(self.eps)
        branch_means = self.running_shat.to(dtype=acc_dtype) / zhat
        aw = self.attention_weights.to(device=self.running_shat.device, dtype=acc_dtype)
        weighted_mean = (branch_means * aw.view(1, 3, 1, 1, 1)).sum(dim=1)
        r = self.current_r.to(device=self.running_shat.device, dtype=acc_dtype)
        return {
            "m": self.running_m.to(dtype=acc_dtype),
            "zhat": zhat,
            "branch_means": branch_means,
            "weighted_mean": weighted_mean,
            "r": r,
            "value_weights": self.value_weights.to(device=self.running_shat.device, dtype=acc_dtype),
            "attention_weights": aw,
        }

    def reduce_tile_for_backward(self, trimmed_output, valid_mask: torch.Tensor | None, global_context):
        if valid_mask is None:
            raise ValueError("StreamingFusedAttentionGeMReducer backward replay requires a valid_mask.")
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 6:
            raise ValueError(f"StreamingFusedAttentionGeMReducer expects payload arity=6, got {len(payload)}")
        y1, y2, y3, logits1, logits2, logits3 = payload
        _validate_value_maps(y1, y2, y3)
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, y1.dtype)
        vw = global_context["value_weights"].to(device=y1.device, dtype=acc_dtype)
        aw = global_context["attention_weights"].to(device=y1.device, dtype=acc_dtype)
        r = global_context["r"].to(device=y1.device, dtype=acc_dtype)
        m = global_context["m"].to(device=y1.device, dtype=acc_dtype)
        zhat = global_context["zhat"].to(device=y1.device, dtype=acc_dtype).clamp_min(self.eps)
        branch_means = global_context["branch_means"].to(device=y1.device, dtype=acc_dtype)
        weighted_mean = global_context["weighted_mean"].to(device=y1.device, dtype=acc_dtype)

        fused_y = vw[0] * y1.to(dtype=acc_dtype) + vw[1] * y2.to(device=y1.device, dtype=acc_dtype) + vw[2] * y3.to(device=y1.device, dtype=acc_dtype)
        x_pow = fused_y.clamp_min(self.eps).pow(r)
        logits = torch.stack(
            [logit.to(device=y1.device, dtype=acc_dtype) for logit in _normalize_three_logits(logits1, logits2, logits3, y1)],
            dim=1,
        )

        valid5d = valid_mask[None, None, None].to(device=y1.device, dtype=torch.bool)
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

        branch_terms = local_s_over_z - branch_means.detach() * local_z_over_z
        scale = (1.0 / r) * weighted_mean.clamp_min(self.eps).pow(1.0 / r - 1.0)
        surrogate = scale.detach() * (branch_terms * aw.view(1, 3, 1, 1, 1)).sum(dim=1)
        return surrogate.to(dtype=y1.dtype)

    def to_reducer(self) -> FusedAttentionGeMReducer:
        reducer = FusedAttentionGeMReducer(
            r_init=float(self.current_r.detach().item()),
            eps=self.eps,
            value_weights=tuple(float(x) for x in self.value_weights.detach().cpu()),
            attention_weights=tuple(float(x) for x in self.attention_weights.detach().cpu()),
            accumulator_dtype=self.accumulator_dtype,
        )
        reducer.r.data.copy_(self.current_r.detach().to(device=reducer.r.device, dtype=reducer.r.dtype))
        reducer.value_weights.data.copy_(self.value_weights.detach().to(device=reducer.value_weights.device, dtype=reducer.value_weights.dtype))
        reducer.attention_weights.data.copy_(self.attention_weights.detach().to(device=reducer.attention_weights.device, dtype=reducer.attention_weights.dtype))
        return reducer

