from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, _new_value_indices, H_DIM, W_DIM


class StreamingGlobalReducerF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(
        ctx,
        inpt: torch.Tensor,
        r: float,
        eps: float,
        grad_lost: Lost,
        seen_indices: Box,
        output_stride: torch.Tensor,
        input_loc: Box,
    ) -> torch.Tensor:
        if inpt.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(inpt.shape)}")

        mean_p_r = inpt.pow(r).mean(dim=(-2, -1), keepdim=True)
        clamped = mean_p_r.clamp_min(eps)
        out = clamped.pow(1.0 / r)

        ctx.save_for_backward(inpt, mean_p_r, clamped)
        ctx.r = r
        ctx.grad_lost = grad_lost
        ctx.seen_indices = seen_indices
        ctx.output_stride = output_stride
        ctx.input_loc = input_loc
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

            input_loc = ctx.input_loc
            if input_loc is not None and input_loc.sides is not None:
                sides = input_loc.sides
                grad_lost = ctx.grad_lost

                lost_top = grad_lost.top if not sides.top else 0
                lost_bottom = grad_lost.bottom if not sides.bottom else 0
                lost_left = grad_lost.left if not sides.left else 0
                lost_right = grad_lost.right if not sides.right else 0

                valid_grad = grad_input[
                    :,
                    :,
                    lost_top : grad_input.shape[H_DIM] - lost_bottom,
                    lost_left : grad_input.shape[W_DIM] - lost_right,
                ]

                data_loc_y = int(input_loc.y // int(ctx.output_stride[1])) + lost_top
                data_loc_x = int(input_loc.x // int(ctx.output_stride[2])) + lost_left
                data_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)

                new_box, updated_total_indices = _new_value_indices(valid_grad.shape, data_loc, ctx.seen_indices)

                ctx.seen_indices.y = updated_total_indices.y
                ctx.seen_indices.height = updated_total_indices.height
                ctx.seen_indices.x = updated_total_indices.x
                ctx.seen_indices.width = updated_total_indices.width
                ctx.seen_indices.sides = updated_total_indices.sides

                deduped_valid_grad = torch.zeros_like(valid_grad)
                if new_box.height > 0 and new_box.width > 0:
                    deduped_valid_grad[
                        :,
                        :,
                        new_box.y : new_box.y + new_box.height,
                        new_box.x : new_box.x + new_box.width,
                    ] = valid_grad[
                        :,
                        :,
                        new_box.y : new_box.y + new_box.height,
                        new_box.x : new_box.x + new_box.width,
                    ]

                grad_input = torch.zeros_like(grad_input)
                grad_input[
                    :,
                    :,
                    lost_top : grad_input.shape[H_DIM] - lost_bottom,
                    lost_left : grad_input.shape[W_DIM] - lost_right,
                ] = deduped_valid_grad

        return grad_input, None, None, None, None, None, None


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
        return streaming_global_reduce(
            logits,
            self.r,
            self.eps,
            self.grad_lost,
            self.seen_indices,
            self.output_stride,
            self.input_loc,
        )
