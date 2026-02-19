import math
import torch

from torch import Tensor
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, H_DIM, W_DIM, _new_value_indices


class StreamingGlobalReducerTileF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(ctx, logits, r, grad_lost, seen_indices, output_stride, input_loc, data_loc):
        probs = torch.sigmoid(logits)

        if input_loc is None or input_loc.sides is None:
            mask = torch.ones_like(probs)
            sum_p_r = probs.pow(r).sum(dim=(-2, -1))
            count = probs.shape[H_DIM] * probs.shape[W_DIM]
            ctx.save_for_backward(probs, mask)
            ctx.r = r
            return sum_p_r, torch.tensor(float(count), device=logits.device, dtype=logits.dtype)

        stride_y = float(output_stride[1].item()) if isinstance(output_stride, torch.Tensor) else float(output_stride[1])
        stride_x = float(output_stride[2].item()) if isinstance(output_stride, torch.Tensor) else float(output_stride[2])

        sides = input_loc.sides
        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0

        valid_h_start = lost_top
        valid_h_end = probs.shape[H_DIM] - lost_bottom
        valid_w_start = lost_left
        valid_w_end = probs.shape[W_DIM] - lost_right

        valid_shape = (
            probs.shape[0],
            probs.shape[1],
            max(valid_h_end - valid_h_start, 0),
            max(valid_w_end - valid_w_start, 0),
        )

        if valid_shape[2] <= 0 or valid_shape[3] <= 0:
            mask = torch.zeros_like(probs)
            sum_p_r = torch.zeros((probs.shape[0], probs.shape[1]), dtype=probs.dtype, device=probs.device)
            ctx.save_for_backward(probs, mask)
            ctx.r = r
            return sum_p_r, torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        if data_loc is None:
            eps = 1e-9
            data_loc_y = int(math.floor((float(input_loc.y) / stride_y) + eps)) + lost_top
            data_loc_x = int(math.floor((float(input_loc.x) / stride_x) + eps)) + lost_left
            tile_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)
        else:
            tile_loc = data_loc

        new_output_box, updated_total_indices = _new_value_indices(valid_shape, tile_loc, seen_indices)

        seen_indices.y = updated_total_indices.y
        seen_indices.height = updated_total_indices.height
        seen_indices.x = updated_total_indices.x
        seen_indices.width = updated_total_indices.width
        seen_indices.sides = updated_total_indices.sides

        mask = torch.zeros_like(probs)
        if new_output_box.height > 0 and new_output_box.width > 0:
            y0 = valid_h_start + new_output_box.y
            y1 = y0 + new_output_box.height
            x0 = valid_w_start + new_output_box.x
            x1 = x0 + new_output_box.width
            mask[:, :, y0:y1, x0:x1] = 1.0

        unseen_probs = probs * mask
        sum_p_r = unseen_probs.pow(r).sum(dim=(-2, -1))
        count = int(mask.sum().item() // (probs.shape[0] * probs.shape[1]))

        ctx.save_for_backward(probs, mask)
        ctx.r = r
        return sum_p_r, torch.tensor(float(count), device=logits.device, dtype=logits.dtype)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_sum_p_r, grad_count):
        probs, mask = ctx.saved_tensors
        r = ctx.r

        grad_logits = None
        if grad_sum_p_r is not None:
            scale = grad_sum_p_r.unsqueeze(-1).unsqueeze(-1)
            grad_logits = scale * r * probs.pow(r - 1.0) * probs * (1.0 - probs) * mask

        return grad_logits, None, None, None, None, None, None


streaming_reducer_tile = StreamingGlobalReducerTileF.apply


class StreamingGlobalReducer(torch.nn.Module):
    """Streaming version of the generalized-mean global reducer.

    This module expects NCHW logits and computes:
        g = (mean(sigmoid(logits) ** r)) ** (1 / r)

    In streaming mode it only accumulates unseen spatial regions per tile.
    """

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
        self._sum_compensation = None
        self._count = 0
        self.data_loc = None

    def forward(self, logits: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(logits.shape)}")

        sum_p_r, count = streaming_reducer_tile(
            logits,
            self.r,
            self.lost,
            self.seen_indices,
            self.output_stride,
            self.input_loc,
            self.data_loc,
        )
        count_value = int(count.item())

        if self._sum_p_r is None:
            self._sum_p_r = sum_p_r
            self._sum_compensation = torch.zeros_like(sum_p_r)
            self._count = count_value
        else:
            # Kahan-style compensated summation for tile accumulation.
            # This minimizes floating-point drift from summing tiles in a
            # different order than full-image reduction.
            y = sum_p_r - self._sum_compensation
            t = self._sum_p_r + y
            self._sum_compensation = (t - self._sum_p_r) - y
            self._sum_p_r = t
            self._count += count_value

        if self._count <= 0:
            mean_p_r = torch.zeros_like(self._sum_p_r)
        else:
            mean_p_r = self._sum_p_r / float(self._count)

        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)
