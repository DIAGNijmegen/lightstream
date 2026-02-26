import torch
import torch.nn as nn

from torch import Tensor
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, _new_value_indices, H_DIM, W_DIM


class StreamingGlobalReducerF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(
        ctx,
        inpt,
        r,
        eps,
        grad_lost,
        seen_indices,
        output_stride,
        input_loc,
        global_sum_r,
        global_count,
        forward_count,
    ):
        if inpt.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(inpt.shape)}")

        sides = input_loc.sides
        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0

        valid = inpt[
            :,
            :,
            lost_top : inpt.shape[H_DIM] - lost_bottom,
            lost_left : inpt.shape[W_DIM] - lost_right,
        ]

        # During no-grad forward (tile reconstruction), only aggregate new pixels
        # so streamed aggregation matches full-image GeM exactly.
        if not torch.is_grad_enabled():
            data_loc_y = int(input_loc.y // output_stride[1]) + lost_top
            data_loc_x = int(input_loc.x // output_stride[2]) + lost_left
            data_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)
            new_output_box, updated_total_indices = _new_value_indices(valid.shape, data_loc, seen_indices)

            seen_indices.y = updated_total_indices.y
            seen_indices.height = updated_total_indices.height
            seen_indices.x = updated_total_indices.x
            seen_indices.width = updated_total_indices.width
            seen_indices.sides = updated_total_indices.sides

            count = int(new_output_box.height * new_output_box.width)
            forward_count.fill_(count)
            if count == 0:
                return torch.zeros((inpt.shape[0], inpt.shape[1], 1, 1), dtype=inpt.dtype, device=inpt.device)

            valid = valid[
                :,
                :,
                new_output_box.y : new_output_box.y + new_output_box.height,
                new_output_box.x : new_output_box.x + new_output_box.width,
            ]
        else:
            forward_count.fill_(int(valid.shape[H_DIM] * valid.shape[W_DIM]))

        mean_p_r = valid.pow(r).mean(dim=(-2, -1), keepdim=True)
        out = mean_p_r.clamp_min(eps).pow(1.0 / r)

        ctx.save_for_backward(inpt, global_sum_r, global_count)
        ctx.r = r
        ctx.eps = eps
        ctx.grad_lost = grad_lost
        ctx.seen_indices = seen_indices
        ctx.output_stride = output_stride
        ctx.input_loc = input_loc
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        inpt, global_sum_r, global_count = ctx.saved_tensors

        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None, None, None, None, None, None

        grad_lost = ctx.grad_lost
        sides = ctx.input_loc.sides
        seen_indices = ctx.seen_indices
        output_stride = ctx.output_stride

        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0

        valid_h = inpt.shape[H_DIM] - lost_top - lost_bottom
        valid_w = inpt.shape[W_DIM] - lost_left - lost_right

        if valid_h <= 0 or valid_w <= 0:
            return torch.zeros_like(inpt), None, None, None, None, None, None, None, None, None, None

        valid = inpt[:, :, lost_top : inpt.shape[H_DIM] - lost_bottom, lost_left : inpt.shape[W_DIM] - lost_right]

        r = ctx.r
        eps = ctx.eps

        if int(global_count.item()) > 0:
            denom = global_count.to(valid.dtype)
            global_mean_r = (global_sum_r.to(valid.dtype) / denom).clamp_min(eps)
            global_out = global_mean_r.pow(1.0 / r)
            coeff = grad_output * global_out.pow(1.0 - r) / denom
        else:
            count = float(valid.shape[H_DIM] * valid.shape[W_DIM])
            local_mean_r = valid.pow(r).mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
            local_out = local_mean_r.pow(1.0 / r)
            coeff = grad_output * local_out.pow(1.0 - r) / count

        valid_grad = coeff * valid.clamp_min(eps).pow(r - 1.0)

        data_loc_y = int(ctx.input_loc.y // output_stride[1]) + lost_top
        data_loc_x = int(ctx.input_loc.x // output_stride[2]) + lost_left
        data_loc = Box(data_loc_y, 0, data_loc_x, 0, ctx.input_loc.sides)

        new_output_box, updated_total_indices = _new_value_indices(valid_grad.shape, data_loc, seen_indices)

        seen_indices.y = updated_total_indices.y
        seen_indices.height = updated_total_indices.height
        seen_indices.x = updated_total_indices.x
        seen_indices.width = updated_total_indices.width
        seen_indices.sides = updated_total_indices.sides

        masked_valid_grad = torch.zeros_like(valid_grad)
        if new_output_box.height > 0 and new_output_box.width > 0:
            masked_valid_grad[
                :,
                :,
                new_output_box.y : new_output_box.y + new_output_box.height,
                new_output_box.x : new_output_box.x + new_output_box.width,
            ] = valid_grad[
                :,
                :,
                new_output_box.y : new_output_box.y + new_output_box.height,
                new_output_box.x : new_output_box.x + new_output_box.width,
            ]

        grad_in = torch.zeros_like(inpt)
        grad_in[
            :,
            :,
            lost_top : inpt.shape[H_DIM] - lost_bottom,
            lost_left : inpt.shape[W_DIM] - lost_right,
        ] = masked_valid_grad

        return grad_in, None, None, None, None, None, None, None, None, None, None


streaming_global_reducer = StreamingGlobalReducerF.apply


class StreamingGlobalReducer(nn.Module):
    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        self.r = float(r)
        self.eps = float(eps)
        self.grad_lost = Lost(0, 0, 0, 0)
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.output_stride = torch.tensor([1, 1, 1])
        self.input_loc = Box(0, 0, 0, 0, None)

        self.global_sum_r = torch.zeros(1, 1, 1, 1)
        self.global_count = torch.zeros(1)
        self.forward_count = torch.zeros(1)

    @classmethod
    def from_global_reducer(cls, module: nn.Module) -> "StreamingGlobalReducer":
        return cls(r=float(module.r), eps=float(module.eps))

    def set_global_stats(self, sum_r: Tensor, count: Tensor):
        self.global_sum_r = sum_r.detach()
        self.global_count = count.detach()

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.forward_count.zero_()

    def forward(self, inpt: Tensor) -> Tensor:
        if self.global_sum_r.device != inpt.device or self.global_sum_r.dtype != inpt.dtype:
            self.global_sum_r = self.global_sum_r.to(device=inpt.device, dtype=inpt.dtype)
        if self.global_count.device != inpt.device or self.global_count.dtype != inpt.dtype:
            self.global_count = self.global_count.to(device=inpt.device, dtype=inpt.dtype)
        if self.forward_count.device != inpt.device:
            self.forward_count = self.forward_count.to(device=inpt.device)
        return streaming_global_reducer(
            inpt,
            self.r,
            self.eps,
            self.grad_lost,
            self.seen_indices,
            self.output_stride,
            self.input_loc,
            self.global_sum_r,
            self.global_count,
            self.forward_count,
        )
