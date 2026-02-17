import torch

from torch import Tensor

from lightstream.core.scnn.utils import Box, Lost, H_DIM, W_DIM, _new_value_indices


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
        self.reset()

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self._sum_p_r = None
        self._count = 0

    def _sum_unseen(self, probs: Tensor):
        if self.input_loc is None or self.input_loc.sides is None:
            sum_p_r = probs.pow(self.r).sum(dim=(-2, -1))
            count = probs.shape[H_DIM] * probs.shape[W_DIM]
            return sum_p_r, count

        output_stride = self.output_stride
        stride_y = float(output_stride[1].item()) if isinstance(output_stride, torch.Tensor) else float(output_stride[1])
        stride_x = float(output_stride[2].item()) if isinstance(output_stride, torch.Tensor) else float(output_stride[2])

        sides = self.input_loc.sides
        lost = self.lost
        lost_top = lost.top if not sides.top else 0
        lost_bottom = lost.bottom if not sides.bottom else 0
        lost_left = lost.left if not sides.left else 0
        lost_right = lost.right if not sides.right else 0

        valid_probs = probs[
            :,
            :,
            lost_top : probs.shape[H_DIM] - lost_bottom,
            lost_left : probs.shape[W_DIM] - lost_right,
        ]

        data_loc_y = int(self.input_loc.y // stride_y) + lost_top
        data_loc_x = int(self.input_loc.x // stride_x) + lost_left
        data_loc = Box(data_loc_y, 0, data_loc_x, 0, self.input_loc.sides)

        new_output_box, updated_total_indices = _new_value_indices(valid_probs.shape, data_loc, self.seen_indices)
        self.seen_indices = updated_total_indices

        if new_output_box.height <= 0 or new_output_box.width <= 0:
            return torch.zeros((probs.shape[0], probs.shape[1]), dtype=probs.dtype, device=probs.device), 0

        unseen = valid_probs[
            :,
            :,
            new_output_box.y : new_output_box.y + new_output_box.height,
            new_output_box.x : new_output_box.x + new_output_box.width,
        ]
        sum_p_r = unseen.pow(self.r).sum(dim=(-2, -1))
        count = unseen.shape[H_DIM] * unseen.shape[W_DIM]
        return sum_p_r, count

    def forward(self, logits: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(logits.shape)}")

        probs = torch.sigmoid(logits)
        sum_p_r, count = self._sum_unseen(probs)

        if self._sum_p_r is None:
            self._sum_p_r = sum_p_r
            self._count = count
        else:
            self._sum_p_r = self._sum_p_r + sum_p_r
            self._count += count

        if self._count <= 0:
            mean_p_r = torch.zeros_like(self._sum_p_r)
        else:
            mean_p_r = self._sum_p_r / float(self._count)

        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)
