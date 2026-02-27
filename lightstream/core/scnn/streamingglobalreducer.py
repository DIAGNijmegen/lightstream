from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost


class StreamingGlobalReducerF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(ctx, inpt: torch.Tensor, r: float, eps: float) -> torch.Tensor:
        if inpt.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(inpt.shape)}")

        mean_p_r = inpt.pow(r).mean(dim=(-2, -1), keepdim=True)
        clamped = mean_p_r.clamp_min(eps)
        out = clamped.pow(1.0 / r)

        ctx.save_for_backward(inpt, mean_p_r, clamped)
        ctx.r = r
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        inpt, mean_p_r, clamped = ctx.saved_tensors
        r = ctx.r

        grad_input = None
        if ctx.needs_input_grad[0]:
            n = inpt.shape[-2] * inpt.shape[-1]

            grad_mean = torch.zeros_like(mean_p_r)
            valid = mean_p_r >= clamped
            grad_mean[valid] = (1.0 / r) * clamped[valid].pow((1.0 / r) - 1.0)

            grad_pow = grad_output * grad_mean / n
            grad_input = grad_pow * (r * inpt.pow(r - 1.0))

        return grad_input, None, None


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
        self.input_loc = Box(0, 0, 0, 0, None)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return streaming_global_reduce(logits, self.r, self.eps)

