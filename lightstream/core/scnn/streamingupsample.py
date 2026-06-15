from __future__ import annotations

import math
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

        pre_output_stride = ctx.pre_upsample_output_stride
        stride_h = int(pre_output_stride[1].item()) if isinstance(pre_output_stride, torch.Tensor) else int(pre_output_stride[1])
        stride_w = int(pre_output_stride[2].item()) if isinstance(pre_output_stride, torch.Tensor) else int(pre_output_stride[2])

        data_loc_y = int(ctx.input_loc.y // stride_h)
        data_loc_x = int(ctx.input_loc.x // stride_w)
        data_loc = Box(data_loc_y, 0, data_loc_x, 0, ctx.input_loc.sides)

        owned_lowres_box, updated_total_indices = _new_value_indices(inpt.shape, data_loc, ctx.seen_indices)

        ctx.seen_indices.y = updated_total_indices.y
        ctx.seen_indices.height = updated_total_indices.height
        ctx.seen_indices.x = updated_total_indices.x
        ctx.seen_indices.width = updated_total_indices.width
        ctx.seen_indices.sides = updated_total_indices.sides

        def _bilinear_backward_support(start: int, length: int, in_size: int, out_size: int) -> tuple[int, int]:
            if length <= 0:
                return 0, 0
            end = start + length
            if out_size <= 0:
                return 0, 0
            scale = float(out_size) / float(max(1, in_size))
            # For align_corners=False, output coordinate o samples input coordinate
            # (o + 0.5) / scale - 0.5. An input index i receives gradients from
            # output locations whose sampling coordinate lies in (i - 1, i + 1).
            # This conservative integer envelope includes all high-resolution
            # locations that can contribute to any owned low-resolution index.
            support_start = math.floor(scale * (float(start) - 0.5) - 0.5) + 1
            support_end = math.ceil(scale * (float(end) + 0.5) - 0.5)
            return max(0, support_start), min(out_size, support_end)

        support_top, support_bottom = _bilinear_backward_support(
            owned_lowres_box.y, owned_lowres_box.height, inpt.shape[H_DIM], H
        )
        support_left, support_right = _bilinear_backward_support(
            owned_lowres_box.x, owned_lowres_box.width, inpt.shape[W_DIM], W
        )

        support_top = max(support_top, lost_top)
        support_bottom = min(support_bottom, H - lost_bottom)
        support_left = max(support_left, lost_left)
        support_right = min(support_right, W - lost_right)

        grad_for_interp = torch.zeros_like(grad_output)
        if support_bottom > support_top and support_right > support_left:
            grad_for_interp[:, :, support_top:support_bottom, support_left:support_right] = grad_output[
                :, :, support_top:support_bottom, support_left:support_right
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

        if grad_in is not None:
            masked_grad_in = torch.zeros_like(grad_in)
            if owned_lowres_box.height > 0 and owned_lowres_box.width > 0:
                masked_grad_in[
                    :,
                    :,
                    owned_lowres_box.y : owned_lowres_box.y + owned_lowres_box.height,
                    owned_lowres_box.x : owned_lowres_box.x + owned_lowres_box.width,
                ] = grad_in[
                    :,
                    :,
                    owned_lowres_box.y : owned_lowres_box.y + owned_lowres_box.height,
                    owned_lowres_box.x : owned_lowres_box.x + owned_lowres_box.width,
                ]
            grad_in = masked_grad_in

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
        self.scale_factor_hw = None
        self.pre_upsample_output_stride = torch.tensor([1, 1, 1])
        self.output_stride = torch.tensor([1, 1, 1])
        self.post_upsample_output_stride = self.output_stride
        self.side_aware_grad_lost = None
        self.backward_valid_lost = None
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
