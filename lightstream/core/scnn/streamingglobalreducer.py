from __future__ import annotations

import torch
import torch.nn as nn

from lightstream.core.scnn.utils import Lost, H_DIM, W_DIM


class StreamingGlobalReducer(nn.Module):
    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        self.r = float(r)
        self.eps = float(eps)
        self.register_buffer("sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("count", torch.tensor(0.0), persistent=False)
        self.register_buffer("covered", torch.zeros((1, 1), dtype=torch.bool), persistent=False)
        self.grad_lost = Lost(0, 0, 0, 0)
        self.input_lost = Lost(0, 0, 0, 0)
        self.output_stride = torch.tensor([1, 1, 1])
        self.reset()

    def reset(self):
        self.input_loc = None
        self.sum = self.sum.new_zeros((1, 1))
        self.count = self.count.new_zeros((1, 1))
        self.covered = self.covered.new_zeros((1, 1), dtype=torch.bool)
        self.min_y = None
        self.min_x = None
        self.max_y = None
        self.max_x = None

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

    def _ensure_coverage(self, y1: int, x1: int, device: torch.device):
        old_h, old_w = int(self.covered.shape[0]), int(self.covered.shape[1])
        if self.covered.device != device:
            self.covered = self.covered.to(device=device)
        if y1 <= old_h and x1 <= old_w:
            return
        new_h = max(old_h, y1)
        new_w = max(old_w, x1)
        new_cov = torch.zeros((new_h, new_w), dtype=torch.bool, device=device)
        new_cov[:old_h, :old_w] = self.covered[:old_h, :old_w]
        self.covered = new_cov

    def _valid_bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int, int]:
        if self.input_loc is None or self.input_loc.sides is None:
            return x, 0, int(x.shape[H_DIM]), 0, int(x.shape[W_DIM])

        sides = self.input_loc.sides
        lost_top = self.input_lost.top if not sides.top else 0
        lost_bottom = self.input_lost.bottom if not sides.bottom else 0
        lost_left = self.input_lost.left if not sides.left else 0
        lost_right = self.input_lost.right if not sides.right else 0

        rel_y0 = int(lost_top)
        rel_y1 = int(x.shape[H_DIM] - lost_bottom)
        rel_x0 = int(lost_left)
        rel_x1 = int(x.shape[W_DIM] - lost_right)

        if rel_y1 <= rel_y0 or rel_x1 <= rel_x0:
            return x[:, :, :0, :0], 0, 0, 0, 0

        valid = x[:, :, rel_y0:rel_y1, rel_x0:rel_x1]

        stride_h = int(self.output_stride[1]) if isinstance(self.output_stride, torch.Tensor) else 1
        stride_w = int(self.output_stride[2]) if isinstance(self.output_stride, torch.Tensor) else 1
        stride_h = max(1, stride_h)
        stride_w = max(1, stride_w)

        abs_y0 = int(self.input_loc.y // stride_h) + rel_y0
        abs_x0 = int(self.input_loc.x // stride_w) + rel_x0
        abs_y1 = abs_y0 + int(valid.shape[H_DIM])
        abs_x1 = abs_x0 + int(valid.shape[W_DIM])
        return valid, abs_y0, abs_y1, abs_x0, abs_x1

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(x.shape)}")

        reduced = x.pow(self.r)
        valid, abs_y0, abs_y1, abs_x0, abs_x1 = self._valid_bounds(reduced)
        self._ensure_state(reduced)

        if valid.shape[H_DIM] > 0 and valid.shape[W_DIM] > 0:
            self._ensure_coverage(abs_y1, abs_x1, valid.device)
            covered_view = self.covered[abs_y0:abs_y1, abs_x0:abs_x1]
            new_mask = ~covered_view

            if torch.any(new_mask):
                selected = valid[:, :, new_mask]
                tile_sum = selected.sum(dim=-1)
                n_new = int(new_mask.sum().item())
                tile_count = torch.full_like(tile_sum, float(n_new))
                self.sum = self.sum + tile_sum
                self.count = self.count + tile_count
                covered_view |= new_mask

            self.min_y = abs_y0 if self.min_y is None else min(self.min_y, abs_y0)
            self.min_x = abs_x0 if self.min_x is None else min(self.min_x, abs_x0)
            self.max_y = abs_y1 if self.max_y is None else max(self.max_y, abs_y1)
            self.max_x = abs_x1 if self.max_x is None else max(self.max_x, abs_x1)

        mean_p_r = self.sum / self.count.clamp_min(1.0)
        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)[:, :, None, None]

    def get_coverage_stats(self) -> dict[str, float]:
        if self.min_y is None or self.min_x is None or self.max_y is None or self.max_x is None:
            return {"covered": 0.0, "bbox": 0.0, "ratio": 1.0}

        y0, y1 = int(self.min_y), int(self.max_y)
        x0, x1 = int(self.min_x), int(self.max_x)
        bbox = max(0, y1 - y0) * max(0, x1 - x0)
        covered = int(self.covered[y0:y1, x0:x1].sum().item()) if bbox > 0 else 0
        ratio = 1.0 if bbox == 0 else float(covered) / float(bbox)
        return {"covered": float(covered), "bbox": float(bbox), "ratio": float(ratio)}

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.aggregate(logits)

    @classmethod
    def from_global_reducer(cls, module: nn.Module) -> "StreamingGlobalReducer":
        return cls(r=float(module.r), eps=float(module.eps))

    def to_global_reducer(self) -> nn.Module:
        from lightstream.models.segment.globalreducer import GlobalReducer

        return GlobalReducer(r=self.r, eps=self.eps)
