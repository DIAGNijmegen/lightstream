from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, _new_value_indices, H_DIM, W_DIM


class StreamingUpsample2dF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(
        ctx,
        inpt,
        size,
        scale_factor,
        mode,
        align_corners,
        recompute_scale_factor,
        grad_lost,
        seen_indices,
        pre_upsample_output_stride,
        output_stride,
        input_loc,
    ):
        ctx.save_for_backward(inpt)
        ctx.size = size
        ctx.scale_factor = scale_factor
        ctx.mode = mode
        ctx.align_corners = align_corners
        ctx.recompute_scale_factor = recompute_scale_factor
        ctx.grad_lost = grad_lost
        ctx.seen_indices = seen_indices
        ctx.pre_upsample_output_stride = pre_upsample_output_stride
        ctx.output_stride = output_stride
        ctx.input_loc = input_loc
        return F.interpolate(
            inpt,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
        )

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (inpt,) = ctx.saved_tensors
        sides = ctx.input_loc.sides
        grad_lost = ctx.grad_lost

        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0

        H = grad_output.shape[H_DIM]
        W = grad_output.shape[W_DIM]
        valid_grad = grad_output[:, :, lost_top : H - lost_bottom, lost_left : W - lost_right]

        output_stride = ctx.output_stride
        stride_h = int(output_stride[1].item()) if isinstance(output_stride, torch.Tensor) else int(output_stride[1])
        stride_w = int(output_stride[2].item()) if isinstance(output_stride, torch.Tensor) else int(output_stride[2])
        # grad_output is in post-upsample coordinates, so locating valid_grad for
        # _new_value_indices(valid_grad.shape, ...) must use the post-upsample stride.
        data_loc_y = int(ctx.input_loc.y // stride_h) + lost_top
        data_loc_x = int(ctx.input_loc.x // stride_w) + lost_left
        data_loc = Box(data_loc_y, 0, data_loc_x, 0, ctx.input_loc.sides)

        new_output_box, updated_total_indices = _new_value_indices(valid_grad.shape, data_loc, ctx.seen_indices)

        ctx.seen_indices.y = updated_total_indices.y
        ctx.seen_indices.height = updated_total_indices.height
        ctx.seen_indices.x = updated_total_indices.x
        ctx.seen_indices.width = updated_total_indices.width
        ctx.seen_indices.sides = updated_total_indices.sides

        grad_for_interp = torch.zeros_like(grad_output)
        if new_output_box.height > 0 and new_output_box.width > 0:
            grad_for_interp[
                :,
                :,
                lost_top + new_output_box.y : lost_top + new_output_box.y + new_output_box.height,
                lost_left + new_output_box.x : lost_left + new_output_box.x + new_output_box.width,
            ] = valid_grad[
                :,
                :,
                new_output_box.y : new_output_box.y + new_output_box.height,
                new_output_box.x : new_output_box.x + new_output_box.width,
            ]

        if ctx.needs_input_grad[0]:
            with torch.enable_grad():
                inpt_grad = inpt.detach().requires_grad_(True)
                out = F.interpolate(
                    inpt_grad,
                    size=ctx.size,
                    scale_factor=ctx.scale_factor,
                    mode=ctx.mode,
                    align_corners=ctx.align_corners,
                    recompute_scale_factor=ctx.recompute_scale_factor,
                )
                grad_in = torch.autograd.grad(out, inpt_grad, grad_for_interp, retain_graph=False, allow_unused=False)[0]
        else:
            grad_in = None

        return grad_in, None, None, None, None, None, None, None, None, None, None


upsample2d = StreamingUpsample2dF.apply


class StreamingUpsample2d(nn.Module):
    _SUPPORTED_MODES = {"nearest", "bilinear"}

    def __init__(
        self,
        size: Optional[tuple[int, int] | int] = None,
        scale_factor: Optional[tuple[float, float] | float] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ):
        super().__init__()

        if size is None and scale_factor is None:
            raise ValueError("StreamingUpsample2d expects either `size` or `scale_factor`.")
        if mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported upsample mode `{mode}` for StreamingUpsample2d. Supported modes: {sorted(self._SUPPORTED_MODES)}"
            )
        if mode == "bilinear" and align_corners not in (None, False):
            raise ValueError("StreamingUpsample2d currently supports bilinear mode only with align_corners=False.")

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

        self.grad_lost = Lost(0, 0, 0, 0)
        self.pre_upsample_output_stride = torch.tensor([1, 1, 1])
        self.output_stride = torch.tensor([1, 1, 1])
        self.reset()

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.input_loc = Box(0, 0, 0, 0, None)

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return upsample2d(
            inpt,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            self.recompute_scale_factor,
            self.grad_lost,
            self.seen_indices,
            self.pre_upsample_output_stride,
            self.output_stride,
            self.input_loc,
        )

    @classmethod
    def from_torch_upsample(cls, module: nn.Upsample) -> "StreamingUpsample2d":
        return cls(
            size=module.size,
            scale_factor=module.scale_factor,
            mode=module.mode,
            align_corners=module.align_corners,
            recompute_scale_factor=module.recompute_scale_factor,
        )

    def to_torch_upsample(self) -> nn.Upsample:
        return nn.Upsample(
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )
