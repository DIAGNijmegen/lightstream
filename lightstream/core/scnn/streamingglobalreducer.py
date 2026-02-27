import torch
import torch.nn as nn

from torch import Tensor

from lightstream.core.scnn.utils import Box, Lost, _new_value_indices, H_DIM, W_DIM


class StreamingGlobalReducerF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inpt: Tensor,
        r: float,
        eps: float,
        grad_lost: Lost,
        seen_indices: Box,
        output_stride: Tensor,
        input_loc: Box,
        running_sum: Tensor,
        running_count: Tensor,
        backward_count: Tensor,
    ):
        sides = input_loc.sides
        lost_top = grad_lost.top if (sides is not None and not sides.top) else 0
        lost_bottom = grad_lost.bottom if (sides is not None and not sides.bottom) else 0
        lost_left = grad_lost.left if (sides is not None and not sides.left) else 0
        lost_right = grad_lost.right if (sides is not None and not sides.right) else 0

        valid = inpt[:, :, lost_top : inpt.shape[H_DIM] - lost_bottom, lost_left : inpt.shape[W_DIM] - lost_right]

        data_loc_y = int(input_loc.y // int(output_stride[1])) + lost_top
        data_loc_x = int(input_loc.x // int(output_stride[2])) + lost_left
        data_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)
        new_box, updated_total_indices = _new_value_indices(valid.shape, data_loc, seen_indices)

        seen_indices.y = updated_total_indices.y
        seen_indices.height = updated_total_indices.height
        seen_indices.x = updated_total_indices.x
        seen_indices.width = updated_total_indices.width
        seen_indices.sides = updated_total_indices.sides

        contrib = torch.zeros_like(inpt)
        contrib_mask = torch.zeros_like(inpt, dtype=torch.bool)
        if new_box.height > 0 and new_box.width > 0:
            rel = valid[
                :,
                :,
                new_box.y : new_box.y + new_box.height,
                new_box.x : new_box.x + new_box.width,
            ]
            rel_p = rel.pow(r)
            running_sum.add_(rel_p.sum(dim=(-2, -1), keepdim=True))
            running_count.add_(float(rel.shape[-2] * rel.shape[-1]))

            y0 = lost_top + new_box.y
            y1 = y0 + new_box.height
            x0 = lost_left + new_box.x
            x1 = x0 + new_box.width
            contrib[:, :, y0:y1, x0:x1].copy_(rel)
            contrib_mask[:, :, y0:y1, x0:x1] = True

        mean = running_sum / running_count.clamp_min(1.0)
        out = mean.clamp_min(eps).pow(1.0 / r)

        ctx.r = r
        ctx.eps = eps
        ctx.input_shape = inpt.shape
        ctx.save_for_backward(contrib, contrib_mask, out, backward_count)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        contrib, contrib_mask, out, backward_count = ctx.saved_tensors
        r = ctx.r
        eps = ctx.eps

        grad_in = torch.zeros(ctx.input_shape, dtype=contrib.dtype, device=contrib.device)
        if contrib_mask.any():
            factor = out.clamp_min(eps).pow(1.0 - r) / backward_count.clamp_min(1.0)
            grad_pixels = grad_output * factor * contrib.clamp_min(eps).pow(r - 1.0)
            grad_in[contrib_mask] = grad_pixels[contrib_mask]

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
        self.output_stride = torch.tensor([1, 1, 1])
        self.reset()

    def reset(self, keep_backward_state: bool = False):
        self.input_loc = Box(0, 0, 0, 0, None)
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.running_sum = None
        self.running_count = None
        if not keep_backward_state:
            self.backward_count = None

    def _ensure_buffers(self, inpt: Tensor):
        shape = (inpt.shape[0], inpt.shape[1], 1, 1)
        if self.running_sum is None or self.running_sum.shape != shape or self.running_sum.device != inpt.device:
            self.running_sum = torch.zeros(shape, dtype=inpt.dtype, device=inpt.device)
            self.running_count = torch.zeros(shape, dtype=inpt.dtype, device=inpt.device)

        if self.backward_count is None or self.backward_count.shape != shape or self.backward_count.device != inpt.device:
            self.backward_count = torch.ones(shape, dtype=inpt.dtype, device=inpt.device)

    def finalize_forward_state(self):
        if self.running_count is None:
            return
        self.backward_count = self.running_count.detach().clone()

    def forward(self, inpt: Tensor) -> Tensor:
        self._ensure_buffers(inpt)
        return streaming_global_reducer(
            inpt,
            self.r,
            self.eps,
            self.grad_lost,
            self.seen_indices,
            self.output_stride,
            self.input_loc,
            self.running_sum,
            self.running_count,
            self.backward_count,
        )

    @classmethod
    def from_global_reducer(cls, module) -> "StreamingGlobalReducer":
        return cls(r=module.r, eps=module.eps)
