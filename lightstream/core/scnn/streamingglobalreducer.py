from __future__ import annotations

import torch
import torch.nn as nn

from lightstream.core.scnn.utils import Box, Lost, _new_value_indices, H_DIM, W_DIM


class StreamingGlobalReducer(nn.Module):
    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        self.r = float(r)
        self.eps = float(eps)
        self.register_buffer("sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("count", torch.tensor(0.0), persistent=False)
        self.grad_lost = Lost(0, 0, 0, 0)
        self.input_lost = Lost(0, 0, 0, 0)
        self.output_stride = torch.tensor([1, 1, 1])
        self.reset()

    def reset(self):
        self.input_loc = None
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.sum = self.sum.new_zeros((1, 1))
        self.count = self.count.new_zeros((1, 1))

    def _ensure_state(self, x: torch.Tensor):
        if (
            self.sum.device != x.device
            or self.sum.dtype != x.dtype
            or self.sum.ndim != 2
            or self.sum.shape[0] != x.shape[0]
            or self.sum.shape[1] != x.shape[1]
        ):
            self.sum = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
            self.count = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)

    def _valid_reducer_input(self, x: torch.Tensor) -> tuple[torch.Tensor, Box]:
        if self.input_loc is None or self.input_loc.sides is None:
            data_loc = Box(0, 0, 0, 0, None)
            return x, data_loc

        sides = self.input_loc.sides
        lost_top = self.input_lost.top if not sides.top else 0
        lost_bottom = self.input_lost.bottom if not sides.bottom else 0
        lost_left = self.input_lost.left if not sides.left else 0
        lost_right = self.input_lost.right if not sides.right else 0

        y0 = int(lost_top)
        y1 = int(x.shape[H_DIM] - lost_bottom)
        x0 = int(lost_left)
        x1 = int(x.shape[W_DIM] - lost_right)

        if y1 <= y0 or x1 <= x0:
            valid = x[:, :, :0, :0]
        else:
            valid = x[:, :, y0:y1, x0:x1]

        stride_h = int(self.output_stride[1]) if isinstance(self.output_stride, torch.Tensor) else 1
        stride_w = int(self.output_stride[2]) if isinstance(self.output_stride, torch.Tensor) else 1
        stride_h = max(1, stride_h)
        stride_w = max(1, stride_w)
        data_loc_y = int(self.input_loc.y // stride_h) + y0
        data_loc_x = int(self.input_loc.x // stride_w) + x0
        data_loc = Box(data_loc_y, 0, data_loc_x, 0, self.input_loc.sides)

        return valid, data_loc

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(x.shape)}")

        reduced = x.pow(self.r)
        valid, data_loc = self._valid_reducer_input(reduced)
        self._ensure_state(reduced)

        if valid.shape[H_DIM] > 0 and valid.shape[W_DIM] > 0:
            new_output_box, updated_total_indices = _new_value_indices(valid.shape, data_loc, self.seen_indices)
            self.seen_indices = updated_total_indices
            if new_output_box.height > 0 and new_output_box.width > 0:
                unique_valid = valid[
                    :,
                    :,
                    new_output_box.y : new_output_box.y + new_output_box.height,
                    new_output_box.x : new_output_box.x + new_output_box.width,
                ]
                tile_sum = unique_valid.sum(dim=(-2, -1))
                tile_count = torch.full_like(tile_sum, float(unique_valid.shape[-2] * unique_valid.shape[-1]))
                self.sum = self.sum + tile_sum
                self.count = self.count + tile_count

        mean_p_r = self.sum / self.count.clamp_min(1.0)
        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)[:, :, None, None]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.aggregate(logits)

    @classmethod
    def from_global_reducer(cls, module: nn.Module) -> "StreamingGlobalReducer":
        return cls(r=float(module.r), eps=float(module.eps))

    def to_global_reducer(self) -> nn.Module:
        from lightstream.models.segment.globalreducer import GlobalReducer

        return GlobalReducer(r=self.r, eps=self.eps)
