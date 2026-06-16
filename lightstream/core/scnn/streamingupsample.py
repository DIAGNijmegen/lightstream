from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, H_DIM, W_DIM



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
        backward_valid_lost,
        upsample_backward_input_lost,
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
        ctx.backward_valid_lost = backward_valid_lost
        ctx.upsample_backward_input_lost = upsample_backward_input_lost
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
        upsample_backward_input_lost = (
            ctx.upsample_backward_input_lost
            if ctx.mode == "bilinear" and ctx.upsample_backward_input_lost is not None
            else ctx.backward_valid_lost or Lost(0, 0, 0, 0)
        )

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
                grad_in = torch.autograd.grad(out, inpt_grad, grad_output, retain_graph=False, allow_unused=False)[0]
        else:
            grad_in = None

        if grad_in is not None:
            # Upsample backward maps a high-resolution gradient tile onto the
            # low-resolution input lattice.  Statistics gathering computes how
            # much of that low-resolution lattice lacks complete support for
            # each border, so crop/zero in low-resolution coordinates here.
            # Do not reuse the high-resolution grad_output loss directly: its
            # units differ from grad_in after interpolation backward.
            input_lost_top = upsample_backward_input_lost.top if not sides.top else 0
            input_lost_bottom = upsample_backward_input_lost.bottom if not sides.bottom else 0
            input_lost_left = upsample_backward_input_lost.left if not sides.left else 0
            input_lost_right = upsample_backward_input_lost.right if not sides.right else 0

            h_end = grad_in.shape[H_DIM] - input_lost_bottom
            w_end = grad_in.shape[W_DIM] - input_lost_right
            masked_grad_in = torch.zeros_like(grad_in)
            if h_end > input_lost_top and w_end > input_lost_left:
                masked_grad_in[:, :, input_lost_top:h_end, input_lost_left:w_end] = grad_in[
                    :, :, input_lost_top:h_end, input_lost_left:w_end
                ]
            grad_in = masked_grad_in

        return grad_in, None, None, None, None, None, None, None, None, None, None, None, None


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
        if mode == "nearest" and align_corners is not None:
            raise ValueError("StreamingUpsample2d requires align_corners=None for nearest mode.")
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
        self.backward_valid_lost = Lost(0, 0, 0, 0)
        self.upsample_backward_input_lost = None
        self.upsample_forward_output_lost = None
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
            self.backward_valid_lost,
            self.upsample_backward_input_lost,
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
