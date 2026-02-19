import math
import torch

from torch import Tensor
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, H_DIM, W_DIM, _new_value_indices


class StreamingGlobalReducerF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(ctx, logits, prev_sum, prev_count, r, eps, lost, seen_indices, output_stride, input_loc, data_loc):
        probs = torch.sigmoid(logits)

        mask = torch.zeros_like(probs, dtype=torch.bool)
        count_new = 0

        if input_loc is None or input_loc.sides is None:
            mask[:] = True
            count_new = probs.shape[H_DIM] * probs.shape[W_DIM]
        else:
            sides = input_loc.sides
            lost_top = lost.top if not sides.top else 0
            lost_bottom = lost.bottom if not sides.bottom else 0
            lost_left = lost.left if not sides.left else 0
            lost_right = lost.right if not sides.right else 0

            valid_shape = (
                probs.shape[0],
                probs.shape[1],
                max(0, probs.shape[H_DIM] - lost_top - lost_bottom),
                max(0, probs.shape[W_DIM] - lost_left - lost_right),
            )

            if data_loc is not None:
                cur_data_loc = data_loc
            else:
                stride_y = float(output_stride[1].item()) if isinstance(output_stride, torch.Tensor) else float(output_stride[1])
                stride_x = float(output_stride[2].item()) if isinstance(output_stride, torch.Tensor) else float(output_stride[2])
                eps_floor = 1e-9
                data_loc_y = int(math.floor((float(input_loc.y) / stride_y) + eps_floor)) + lost_top
                data_loc_x = int(math.floor((float(input_loc.x) / stride_x) + eps_floor)) + lost_left
                cur_data_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)

            new_output_box, updated_total_indices = _new_value_indices(valid_shape, cur_data_loc, seen_indices)

            seen_indices.y = updated_total_indices.y
            seen_indices.height = updated_total_indices.height
            seen_indices.x = updated_total_indices.x
            seen_indices.width = updated_total_indices.width
            seen_indices.sides = updated_total_indices.sides

            if new_output_box.height > 0 and new_output_box.width > 0:
                y0 = lost_top + new_output_box.y
                y1 = y0 + new_output_box.height
                x0 = lost_left + new_output_box.x
                x1 = x0 + new_output_box.width
                mask[:, :, y0:y1, x0:x1] = True
                count_new = new_output_box.height * new_output_box.width

        unseen_sum = (probs.pow(r) * mask.to(probs.dtype)).sum(dim=(-2, -1))
        new_sum = prev_sum + unseen_sum
        new_count = int(prev_count) + int(count_new)

        if new_count <= 0:
            output = torch.zeros_like(new_sum)
        else:
            output = (new_sum / float(new_count)).clamp_min(float(eps)).pow(1.0 / float(r))

        ctx.r = float(r)
        ctx.eps = float(eps)
        ctx.new_count = new_count
        ctx.save_for_backward(probs, mask, new_sum)
        return output, new_sum, torch.tensor(float(new_count), device=logits.device, dtype=logits.dtype)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output, grad_new_sum, grad_count_ignored):
        probs, mask, new_sum = ctx.saved_tensors

        if ctx.new_count > 0:
            inv_count = 1.0 / float(ctx.new_count)
            mean_p_r = new_sum * inv_count
            clamp_mask = (mean_p_r >= ctx.eps).to(mean_p_r.dtype)
            dy_dmean = (1.0 / ctx.r) * mean_p_r.clamp_min(ctx.eps).pow((1.0 / ctx.r) - 1.0) * clamp_mask

            grad_new_sum_total = grad_new_sum + (grad_output * dy_dmean * inv_count)
            grad_prev_sum = grad_new_sum_total

            grad_probs = grad_new_sum_total[:, :, None, None] * ctx.r * probs.pow(ctx.r - 1.0)
            grad_probs = grad_probs * mask.to(grad_probs.dtype)
            grad_logits = grad_probs * probs * (1.0 - probs)
        else:
            grad_prev_sum = torch.zeros_like(new_sum)
            grad_logits = torch.zeros_like(probs)

        return grad_logits, grad_prev_sum, None, None, None, None, None, None, None, None


streaming_global_reduce = StreamingGlobalReducerF.apply


class StreamingGlobalReducer(torch.nn.Module):
    """Streaming version of the generalized-mean global reducer."""

    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        self.r = float(r)
        self.eps = float(eps)

        self.grad_lost = None
        self.lost = Lost(0, 0, 0, 0)
        self.output_stride = torch.tensor([1.0, 1.0, 1.0])
        self.input_loc = Box(0, 0, 0, 0, None)
        self.data_loc = None
        self.reset()

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self._sum_p_r = None
        self._count = 0
        self.data_loc = None

    def forward(self, logits: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(logits.shape)}")

        if self._sum_p_r is None:
            self._sum_p_r = torch.zeros((logits.shape[0], logits.shape[1]), dtype=logits.dtype, device=logits.device)

        out, new_sum, new_count = streaming_global_reduce(
            logits,
            self._sum_p_r,
            self._count,
            self.r,
            self.eps,
            self.lost,
            self.seen_indices,
            self.output_stride,
            self.input_loc,
            self.data_loc,
        )
        self._sum_p_r = new_sum
        self._count = int(new_count.detach().item())
        return out
