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
        backward_valid_lost,
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
        backward_valid_lost = ctx.backward_valid_lost or Lost(0, 0, 0, 0)

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

        owned_support_top, owned_support_bottom = _bilinear_backward_support(
            owned_lowres_box.y, owned_lowres_box.height, inpt.shape[H_DIM], H
        )
        owned_support_left, owned_support_right = _bilinear_backward_support(
            owned_lowres_box.x, owned_lowres_box.width, inpt.shape[W_DIM], W
        )
        clipped_support_edges = []
        if owned_support_top < valid_top:
            clipped_support_edges.append("top")
        if owned_support_bottom > valid_bottom:
            clipped_support_edges.append("bottom")
        if owned_support_left < valid_left:
            clipped_support_edges.append("left")
        if owned_support_right > valid_right:
            clipped_support_edges.append("right")

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

        if clipped_support_edges:
            logger.debug(
                "StreamingUpsample2dF.backward found owned low-res cells whose required high-res support "
                "extends beyond valid grad_output bounds: input_loc=%s sides=%s tile_sides=%s "
                "owned_lowres_box=%s required_support=(top=%s, bottom=%s, left=%s, right=%s) "
                "valid_highres=(top=%s, bottom=%s, left=%s, right=%s) complete_lowres_box=%s clipped_edges=%s",
                ctx.input_loc,
                sides,
                owned_lowres_box.sides,
                owned_lowres_box,
                owned_support_top,
                owned_support_bottom,
                owned_support_left,
                owned_support_right,
                valid_top,
                valid_bottom,
                valid_left,
                valid_right,
                complete_lowres_box,
                tuple(clipped_support_edges),
            )

        owned_y1 = owned_lowres_box.y + owned_lowres_box.height
        owned_x1 = owned_lowres_box.x + owned_lowres_box.width
        skipped_top = complete_lowres_box.y - owned_lowres_box.y
        skipped_left = complete_lowres_box.x - owned_lowres_box.x
        skipped_bottom = owned_y1 - (complete_lowres_box.y + complete_lowres_box.height)
        skipped_right = owned_x1 - (complete_lowres_box.x + complete_lowres_box.width)

        dropped_cells = (
            skipped_top * owned_lowres_box.width
            + skipped_bottom * owned_lowres_box.width
            + skipped_left * complete_lowres_box.height
            + skipped_right * complete_lowres_box.height
        )
        if dropped_cells > 0:
            logger.debug(
                "StreamingUpsample2dF.backward dropped candidate low-res cells with incomplete high-res support: "
                "dropped_cells=%s skipped=(top=%s, bottom=%s, left=%s, right=%s) "
                "owned_lowres_box=%s complete_lowres_box=%s data_loc=%s "
                "valid_highres=(top=%s, bottom=%s, left=%s, right=%s)",
                dropped_cells,
                skipped_top,
                skipped_bottom,
                skipped_left,
                skipped_right,
                owned_lowres_box,
                complete_lowres_box,
                data_loc,
                valid_top,
                valid_bottom,
                valid_left,
                valid_right,
            )

        if complete_lowres_box.height > 0 and complete_lowres_box.width > 0:
            # seen_indices is a scan-order frontier and can represent only a contiguous
            # prefix of emitted low-resolution cells. Advance it only over cells that
            # were actually emitted; if the complete box does not start at the owned
            # box origin, leave the frontier unchanged rather than marking a gap seen.
            if complete_lowres_box.y == owned_lowres_box.y and complete_lowres_box.x == owned_lowres_box.x:
                emitted_rel_bottom = complete_lowres_box.y + complete_lowres_box.height
                emitted_rel_right = complete_lowres_box.x + complete_lowres_box.width
                emitted_abs_bottom = data_loc.y + emitted_rel_bottom
                emitted_abs_right = data_loc.x + emitted_rel_right

                ctx.seen_indices.y = updated_total_indices.y
                ctx.seen_indices.height = emitted_abs_bottom if data_loc.x == 0 else updated_total_indices.height
                ctx.seen_indices.x = emitted_abs_right
                ctx.seen_indices.width = updated_total_indices.width
                ctx.seen_indices.sides = updated_total_indices.sides
            else:
                logger.debug(
                    "StreamingUpsample2dF.backward did not advance seen_indices because emitted low-res cells "
                    "do not form a contiguous scan-order prefix: owned_lowres_box=%s complete_lowres_box=%s",
                    owned_lowres_box,
                    complete_lowres_box,
                )

        support_top, support_bottom = _bilinear_backward_support(
            complete_lowres_box.y, complete_lowres_box.height, inpt.shape[H_DIM], H
        )
        support_left, support_right = _bilinear_backward_support(
            complete_lowres_box.x, complete_lowres_box.width, inpt.shape[W_DIM], W
        )

        grad_for_interp = torch.zeros_like(grad_output)
        if valid_bottom > valid_top and valid_right > valid_left:
            grad_for_interp[:, :, valid_top:valid_bottom, valid_left:valid_right] = grad_output[
                :, :, valid_top:valid_bottom, valid_left:valid_right
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
            input_lost_top = backward_valid_lost.top if not sides.top else 0
            input_lost_bottom = backward_valid_lost.bottom if not sides.bottom else 0
            input_lost_left = backward_valid_lost.left if not sides.left else 0
            input_lost_right = backward_valid_lost.right if not sides.right else 0
            masked_grad_in = torch.zeros_like(grad_in)
            if grad_in.shape[H_DIM] > input_lost_top + input_lost_bottom and grad_in.shape[W_DIM] > input_lost_left + input_lost_right:
                masked_grad_in[
                    :,
                    :,
                    input_lost_top : grad_in.shape[H_DIM] - input_lost_bottom,
                    input_lost_left : grad_in.shape[W_DIM] - input_lost_right,
                ] = grad_in[
                    :,
                    :,
                    input_lost_top : grad_in.shape[H_DIM] - input_lost_bottom,
                    input_lost_left : grad_in.shape[W_DIM] - input_lost_right,
                ]
            grad_in = masked_grad_in

        return grad_in, None, None, None, None, None, None, None, None, None, None, None


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
        self.backward_valid_lost = Lost(0, 0, 0, 0)
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
