from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, _new_value_indices


class StreamingGlobalReducerF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(
        ctx,
        inpt: torch.Tensor,
        r: float,
        eps: float,
        global_mean_pow_r: torch.Tensor,
        global_count: int,
        grad_lost: Lost,
        seen_indices: Box,
        output_stride: torch.Tensor,
        input_loc: Box,
    ) -> torch.Tensor:
        if inpt.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(inpt.shape)}")

        clamped = global_mean_pow_r.clamp_min(eps)
        out = clamped.pow(1.0 / r)

        ctx.save_for_backward(inpt, clamped)
        ctx.r = r
        ctx.global_count = int(max(1, global_count))
        ctx.grad_lost = grad_lost
        ctx.seen_indices = seen_indices
        ctx.output_stride = output_stride
        ctx.input_loc = input_loc
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        inpt, clamped = ctx.saved_tensors
        r = ctx.r

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_scale = clamped.pow((1.0 / r) - 1.0) / float(ctx.global_count)
            grad_input = grad_output * grad_scale * inpt.pow(r - 1.0)

        return grad_input, None, None, None, None, None, None, None, None


streaming_global_reduce = StreamingGlobalReducerF.apply


class StreamingGlobalReducer(nn.Module):
    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")

        self.r = float(r)
        self.eps = float(eps)

        self.grad_lost = Lost(0, 0, 0, 0)
        self.output_stride = torch.tensor([1, 1, 1])
        self.reset()

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.forward_seen_indices = Box(0, 0, 0, 0, None)
        self.input_loc = Box(0, 0, 0, 0, None)
        self._sum_pow_r = None
        self._count = 0
        self._cached_mean_pow_r = None

    def _accumulate_unique_forward_sum(self, logits: torch.Tensor) -> None:
        if self.input_loc is None or self.input_loc.sides is None:
            unique = logits
        else:
            data_loc = Box(int(self.input_loc.y), 0, int(self.input_loc.x), 0, self.input_loc.sides)
            new_box, updated = _new_value_indices(logits.shape, data_loc, self.forward_seen_indices)
            self.forward_seen_indices.y = updated.y
            self.forward_seen_indices.height = updated.height
            self.forward_seen_indices.x = updated.x
            self.forward_seen_indices.width = updated.width
            self.forward_seen_indices.sides = updated.sides
            if new_box.height <= 0 or new_box.width <= 0:
                return
            unique = logits[
                :,
                :,
                new_box.y : new_box.y + new_box.height,
                new_box.x : new_box.x + new_box.width,
            ]

        sum_pow = unique.pow(self.r).sum(dim=(-2, -1), keepdim=True)
        count = int(unique.shape[-2] * unique.shape[-1])
        if self._sum_pow_r is None:
            self._sum_pow_r = sum_pow
        else:
            self._sum_pow_r = self._sum_pow_r + sum_pow
        self._count += count
        self._cached_mean_pow_r = (self._sum_pow_r / max(1, self._count)).clamp_min(self.eps)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            self._accumulate_unique_forward_sum(logits)

        mean_pow = self._cached_mean_pow_r
        count = self._count
        if mean_pow is None or count <= 0:
            mean_pow = logits.pow(self.r).mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            count = int(logits.shape[-2] * logits.shape[-1])

        return streaming_global_reduce(
            logits,
            self.r,
            self.eps,
            mean_pow,
            count,
            self.grad_lost,
            self.seen_indices,
            self.output_stride,
            self.input_loc,
        )
