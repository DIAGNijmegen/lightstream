from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, _new_value_indices, H_DIM, W_DIM


logger = logging.getLogger(__name__)


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

        valid_top = lost_top
        valid_bottom = H - lost_bottom
        valid_left = lost_left
        valid_right = W - lost_right

        low_y0 = owned_lowres_box.y
        low_y1 = owned_lowres_box.y + owned_lowres_box.height
        low_x0 = owned_lowres_box.x
        low_x1 = owned_lowres_box.x + owned_lowres_box.width

        while low_y0 < low_y1:
            support_top, _ = _bilinear_backward_support(low_y0, 1, inpt.shape[H_DIM], H)
            if support_top >= valid_top:
                break
            low_y0 += 1

        while low_y0 < low_y1:
            _, support_bottom = _bilinear_backward_support(low_y1 - 1, 1, inpt.shape[H_DIM], H)
            if support_bottom <= valid_bottom:
                break
            low_y1 -= 1

        while low_x0 < low_x1:
            support_left, _ = _bilinear_backward_support(low_x0, 1, inpt.shape[W_DIM], W)
            if support_left >= valid_left:
                break
            low_x0 += 1

        while low_x0 < low_x1:
            _, support_right = _bilinear_backward_support(low_x1 - 1, 1, inpt.shape[W_DIM], W)
            if support_right <= valid_right:
                break
            low_x1 -= 1

        complete_lowres_box = Box(
            low_y0,
            max(0, low_y1 - low_y0),
            low_x0,
            max(0, low_x1 - low_x0),
            owned_lowres_box.sides,
        )

        owned_y1 = owned_lowres_box.y + owned_lowres_box.height
        owned_x1 = owned_lowres_box.x + owned_lowres_box.width
        skipped_top = complete_lowres_box.y - owned_lowres_box.y
        skipped_left = complete_lowres_box.x - owned_lowres_box.x
        skipped_bottom = owned_y1 - (complete_lowres_box.y + complete_lowres_box.height)
        skipped_right = owned_x1 - (complete_lowres_box.x + complete_lowres_box.width)

        if skipped_top > 0 or skipped_left > 0:
            raise RuntimeError(
                "StreamingUpsample2dF.backward would skip low-res cells on the top/left edge, "
                "which cannot be owned by a later tile. "
                f"owned_lowres_box={owned_lowres_box}, complete_lowres_box={complete_lowres_box}, "
                f"data_loc={data_loc}, valid_highres=({valid_top}:{valid_bottom}, {valid_left}:{valid_right})"
            )

        if skipped_bottom > 0 and sides.bottom:
            raise RuntimeError(
                "StreamingUpsample2dF.backward cannot defer bottom low-res cells from the final tile row. "
                f"owned_lowres_box={owned_lowres_box}, complete_lowres_box={complete_lowres_box}, data_loc={data_loc}"
            )
        if skipped_right > 0 and sides.right:
            raise RuntimeError(
                "StreamingUpsample2dF.backward cannot defer right low-res cells from the final tile column. "
                f"owned_lowres_box={owned_lowres_box}, complete_lowres_box={complete_lowres_box}, data_loc={data_loc}"
            )
        if skipped_bottom > 0 or skipped_right > 0:
            logger.debug(
                "Deferring incomplete low-res upsample gradient cells to a later tile: "
                "owned_lowres_box=%s complete_lowres_box=%s data_loc=%s valid_highres=(%s:%s, %s:%s)",
                owned_lowres_box,
                complete_lowres_box,
                data_loc,
                valid_top,
                valid_bottom,
                valid_left,
                valid_right,
            )

        if complete_lowres_box.height > 0 and complete_lowres_box.width > 0:
            emitted_rel_bottom = complete_lowres_box.y + complete_lowres_box.height
            emitted_rel_right = complete_lowres_box.x + complete_lowres_box.width
            emitted_abs_bottom = data_loc.y + emitted_rel_bottom
            emitted_abs_right = data_loc.x + emitted_rel_right

            updated_y = updated_total_indices.y
            updated_height = updated_total_indices.height
            if data_loc.x == 0:
                updated_height = emitted_abs_bottom
            updated_x = emitted_abs_right

            if updated_x > emitted_abs_right or (data_loc.x == 0 and updated_height > emitted_abs_bottom):
                raise RuntimeError(
                    "StreamingUpsample2dF.backward advanced seen_indices past an un-emitted low-res cell. "
                    f"updated=(y={updated_y}, height={updated_height}, x={updated_x}), "
                    f"emitted_abs_bottom={emitted_abs_bottom}, emitted_abs_right={emitted_abs_right}, "
                    f"owned_lowres_box={owned_lowres_box}, complete_lowres_box={complete_lowres_box}"
                )

            ctx.seen_indices.y = updated_y
            ctx.seen_indices.height = updated_height
            ctx.seen_indices.x = updated_x
            ctx.seen_indices.width = updated_total_indices.width
            ctx.seen_indices.sides = updated_total_indices.sides

        support_top, support_bottom = _bilinear_backward_support(
            complete_lowres_box.y, complete_lowres_box.height, inpt.shape[H_DIM], H
        )
        support_left, support_right = _bilinear_backward_support(
            complete_lowres_box.x, complete_lowres_box.width, inpt.shape[W_DIM], W
        )

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
            if complete_lowres_box.height > 0 and complete_lowres_box.width > 0:
                masked_grad_in[
                    :,
                    :,
                    complete_lowres_box.y : complete_lowres_box.y + complete_lowres_box.height,
                    complete_lowres_box.x : complete_lowres_box.x + complete_lowres_box.width,
                ] = grad_in[
                    :,
                    :,
                    complete_lowres_box.y : complete_lowres_box.y + complete_lowres_box.height,
                    complete_lowres_box.x : complete_lowres_box.x + complete_lowres_box.width,
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
