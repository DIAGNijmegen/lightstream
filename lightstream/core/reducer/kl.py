"""Teacher-to-student attention KL reducers for Lightstream.

This file is intentionally self-contained so it can be copied into a downstream
project without registering the reducers in :mod:`lightstream.core.reducer`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer.reducer_base import BaseReducer
from .reducer.utils import prepare_spatial_mask, resolve_accumulator_dtype


def _validate_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> None:
    if student_logits.ndim != 4:
        raise ValueError(f"Expected NCHW attention logits, got shape={tuple(student_logits.shape)}")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Student and teacher logits must have identical [N,C,H,W] shapes; "
            f"got student={tuple(student_logits.shape)}, teacher={tuple(teacher_logits.shape)}."
        )


class AttentionKLDivergenceReducer(BaseReducer):
    """Compute ``KL(normalized-sigmoid(teacher) || softmax(student))``.

    The two inputs are class-wise attention logits with shape ``[N,C,H,W]``.
    Each ``[N,C]`` plane is normalized independently over all valid ``H*W``
    positions. The result has shape ``[N,C,1,1]``: spatial KL terms are summed,
    while averaging over samples/classes is deliberately left to the caller.
    """

    def __init__(
        self,
        teacher_temperature: float = 1.0,
        accumulator_dtype: torch.dtype | None = None,
        mask_resize: bool = False,
        mask_resize_mode: str = "nearest",
    ):
        super().__init__()
        if teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be greater than zero.")
        self.teacher_temperature = float(teacher_temperature)
        self.accumulator_dtype = accumulator_dtype
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None):
        if len(inputs) != 2:
            raise ValueError(
                "AttentionKLDivergenceReducer expects exactly two inputs "
                f"(student_logits, teacher_logits), got {len(inputs)}."
            )
        student_logits, teacher_logits = inputs
        _validate_logits(student_logits, teacher_logits)

        if self._streaming_passthrough:
            payload = (student_logits.view_as(student_logits), teacher_logits.view_as(teacher_logits))
            self._last_inputs, self._last_output = payload, payload[0]
            return payload

        dtype = resolve_accumulator_dtype(self.accumulator_dtype, student_logits.dtype)
        student = student_logits.to(dtype)
        # The EMA teacher is a fixed target. Detaching here still leaves the raw
        # branch connected during SCNN's reducer-passthrough statistics probe.
        teacher = teacher_logits.detach().to(dtype) / self.teacher_temperature
        teacher_log_scores = F.logsigmoid(teacher)  # log(sigmoid(teacher))

        if mask is None:
            valid = torch.ones(
                (student.shape[0], 1, *student.shape[-2:]),
                device=student.device,
                dtype=torch.bool,
            )
        else:
            valid = prepare_spatial_mask(
                mask,
                student_logits,
                mask_resize=self.mask_resize,
                mask_resize_mode=self.mask_resize_mode,
            )

        neg_inf = torch.tensor(float("-inf"), device=student.device, dtype=dtype)
        masked_student = torch.where(valid, student, neg_inf)
        masked_teacher_log_scores = torch.where(valid, teacher_log_scores, neg_inf)
        any_valid = valid.flatten(2).any(dim=-1, keepdim=True).unsqueeze(-1)

        # p is the online spatial softmax and q is the EMA spatially normalized
        # sigmoid. Both normalizations are independent for every [N,C] plane.
        log_p = masked_student - torch.logsumexp(masked_student, dim=(-2, -1), keepdim=True)
        log_q = masked_teacher_log_scores - torch.logsumexp(
            masked_teacher_log_scores, dim=(-2, -1), keepdim=True
        )
        q = torch.where(valid, log_q.exp(), torch.zeros_like(log_q))
        terms = torch.where(valid, q * (log_q - log_p), torch.zeros_like(q))

        # Sum over spatial positions only. Call output.mean() to subsequently
        # average the KL values over batch samples and attention classes.
        kl = terms.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        kl = torch.where(any_valid, kl, torch.zeros_like(kl))
        return kl.to(student_logits.dtype)

    def to_streaming(self) -> "StreamingAttentionKLDivergenceReducer":
        return StreamingAttentionKLDivergenceReducer(
            teacher_temperature=self.teacher_temperature,
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
        )


class StreamingAttentionKLDivergenceReducer(BaseStreamingGlobalReducer):
    """Tiled implementation of :class:`AttentionKLDivergenceReducer`."""

    def __init__(
        self,
        teacher_temperature: float = 1.0,
        accumulator_dtype: torch.dtype | None = None,
        mask_resize: bool = False,
        mask_resize_mode: str = "nearest",
    ):
        super().__init__(mode="sum", accumulator_dtype=accumulator_dtype)
        if teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be greater than zero.")
        self.teacher_temperature = float(teacher_temperature)
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode
        self._output_dtype = torch.float32
        self.register_buffer("teacher_denominator", torch.zeros(0), persistent=False)
        self.register_buffer("teacher_u_log_u", torch.zeros(0), persistent=False)
        self.register_buffer("teacher_student_cross", torch.zeros(0), persistent=False)
        self.register_buffer("student_running_max", torch.zeros(0), persistent=False)
        self.register_buffer("student_exp_sum", torch.zeros(0), persistent=False)

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None):
        if len(inputs) != 2:
            raise ValueError(
                "StreamingAttentionKLDivergenceReducer expects exactly two inputs "
                f"(student_logits, teacher_logits), got {len(inputs)}."
            )
        student_logits, teacher_logits = inputs
        _validate_logits(student_logits, teacher_logits)
        self._last_inputs, self._last_output = (student_logits, teacher_logits), student_logits
        return student_logits, teacher_logits

    def accumulate_stream_tile(self, trimmed_output, tile_y, tile_x, sides, dst_box, user_mask=None):
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 2:
            raise ValueError(f"StreamingAttentionKLDivergenceReducer expects payload arity=2, got {len(payload)}")
        student, teacher = payload
        _validate_logits(student, teacher)
        y0, y1, x0, x1 = dst_box
        seen = self._stream_seen_mask[y0:y1, x0:x1]
        new = ~seen
        effective = new if user_mask is None else new & user_mask.to(device=new.device, dtype=torch.bool)
        if self._debug_replay_enabled:
            if self._replay_assignments is None:
                raise RuntimeError("Reducer replay assignments are not initialized.")
            self._replay_assignments.append(
                (int(tile_y), int(tile_x), bool(sides.top), bool(sides.left), bool(sides.right),
                 bool(sides.bottom), int(student.shape[-2]), int(student.shape[-1]), int(y0),
                 int(y1), int(x0), int(x1), 2)
            )
        if torch.any(effective):
            self.accumulate_valid_tile(payload, effective)
        seen |= new

    def build_backward_pair(self, trimmed_output, gradient, *, input_y, input_x, sides, valid_mask=None):
        payload = self._parse_multi_input_payload(trimmed_output)
        if len(payload) != 2:
            raise ValueError(f"StreamingAttentionKLDivergenceReducer expects payload arity=2, got {len(payload)}")
        student, teacher = payload
        _validate_logits(student, teacher)
        if self._debug_replay_enabled:
            if self._replay_assignments is None or self._replay_cursor is None:
                raise RuntimeError("Reducer replay state is not initialized. Call start_backward_replay() first.")
            self._replay_cursor = self._validate_replay_assignment(
                assignments=self._replay_assignments,
                cursor=self._replay_cursor,
                input_y=input_y,
                input_x=input_x,
                sides=sides,
                expected_h=student.shape[-2],
                expected_w=student.shape[-1],
                expected_arity=2,
            )
        return self.reduce_tile_for_backward(payload, valid_mask, self.extra_state_for_backward()), gradient

    def init_reduction_state(self, *, batch_size, channels, device, dtype, accumulator_dtype):
        shape = (batch_size, channels, 1, 1)
        self._output_dtype = dtype
        self.teacher_denominator = torch.zeros(shape, device=device, dtype=accumulator_dtype)
        self.teacher_u_log_u = torch.zeros(shape, device=device, dtype=accumulator_dtype)
        self.teacher_student_cross = torch.zeros(shape, device=device, dtype=accumulator_dtype)
        self.student_running_max = torch.full(
            shape, float("-inf"), device=device, dtype=accumulator_dtype
        )
        self.student_exp_sum = torch.zeros(shape, device=device, dtype=accumulator_dtype)

    def accumulate_valid_tile(self, tile, valid_mask):
        student_logits, teacher_logits = self._parse_multi_input_payload(tile)
        _validate_logits(student_logits, teacher_logits)
        if self.teacher_denominator.numel() == 0:
            self.reset_stream_state(
                student_logits.shape[0], student_logits.shape[1], student_logits.device, student_logits.dtype
            )
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, student_logits.dtype)
        student = student_logits.detach().to(dtype)
        teacher = teacher_logits.detach().to(dtype) / self.teacher_temperature
        valid = valid_mask[None, None].to(device=student.device, dtype=torch.bool)

        u = torch.where(valid, torch.sigmoid(teacher), torch.zeros_like(teacher))
        log_u = F.logsigmoid(teacher)
        self.teacher_denominator += u.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        self.teacher_u_log_u += (u * log_u).sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        self.teacher_student_cross += (u * student).sum(dim=(-2, -1), keepdim=True, dtype=dtype)

        masked_student = torch.where(valid, student, torch.full_like(student, float("-inf")))
        tile_max = masked_student.amax(dim=(-2, -1), keepdim=True)
        tile_exp = torch.where(valid, torch.exp(masked_student - tile_max), torch.zeros_like(student))
        tile_exp_sum = tile_exp.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        new_max = torch.maximum(self.student_running_max, tile_max)
        old_scale = torch.exp(self.student_running_max - new_max)
        tile_scale = torch.exp(tile_max - new_max)
        self.student_exp_sum = self.student_exp_sum * old_scale + tile_exp_sum * tile_scale
        self.student_running_max = new_max

    def finalize_from_state(self):
        if self.teacher_denominator.numel() == 0:
            raise RuntimeError("Streaming attention KL state is empty.")
        tiny = torch.finfo(self.teacher_denominator.dtype).tiny
        nonempty = self.teacher_denominator > 0
        zq = self.teacher_denominator.clamp_min(tiny)
        zp = self.student_exp_sum.clamp_min(tiny)
        log_zp = self.student_running_max + torch.log(zp)
        # sum_i q_i(log q_i - log p_i)
        kl = (self.teacher_u_log_u - self.teacher_student_cross) / zq - torch.log(zq) + log_zp
        return torch.where(nonempty, kl, torch.zeros_like(kl)).to(self._output_dtype)

    def extra_state_for_backward(self):
        tiny = torch.finfo(self.teacher_denominator.dtype).tiny
        return {
            "teacher_denominator": self.teacher_denominator.clamp_min(tiny).detach(),
            "student_max": self.student_running_max.detach(),
            "student_exp_sum": self.student_exp_sum.clamp_min(tiny).detach(),
            "nonempty": (self.teacher_denominator > 0).detach(),
        }

    def reduce_tile_for_backward(self, trimmed_output, valid_mask, global_context):
        student_logits, teacher_logits = self._parse_multi_input_payload(trimmed_output)
        _validate_logits(student_logits, teacher_logits)
        if valid_mask is None:
            valid_mask = torch.ones(student_logits.shape[-2:], device=student_logits.device, dtype=torch.bool)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, student_logits.dtype)
        student = student_logits.to(dtype)
        teacher = teacher_logits.detach().to(dtype) / self.teacher_temperature
        valid = valid_mask[None, None].to(device=student.device, dtype=torch.bool)

        p = torch.exp(student - global_context["student_max"].to(student.device))
        p = p / global_context["student_exp_sum"].to(student.device)
        q = torch.sigmoid(teacher) / global_context["teacher_denominator"].to(student.device)
        slope = torch.where(valid, p - q, torch.zeros_like(student)).detach()
        slope = torch.where(global_context["nonempty"].to(student.device), slope, torch.zeros_like(slope))

        # This surrogate's derivative with respect to each student logit is
        # exactly p-q, the derivative of KL(q||p). Teacher logits stay detached.
        return streaming_reduce_tile(slope * student, valid_mask, None).to(student_logits.dtype)

    def to_reducer(self) -> AttentionKLDivergenceReducer:
        return AttentionKLDivergenceReducer(
            teacher_temperature=self.teacher_temperature,
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
        )
