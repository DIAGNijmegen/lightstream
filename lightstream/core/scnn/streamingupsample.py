import math
import torch

from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, _new_value_indices, H_DIM, W_DIM


class StreamingUpsampleF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(ctx, inpt, size, scale_factor, mode, align_corners, grad_lost, seen_indices, output_stride, input_loc):
        ctx.size = size
        ctx.scale_factor = scale_factor
        ctx.mode = mode
        ctx.align_corners = align_corners
        ctx.grad_lost = grad_lost
        ctx.seen_indices = seen_indices
        ctx.output_stride = output_stride
        ctx.input_loc = input_loc
        ctx.save_for_backward(inpt)
        return torch.nn.functional.interpolate(
            inpt,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (inpt,) = ctx.saved_tensors
        grad_in = None

        if ctx.needs_input_grad[0]:
            sides = ctx.input_loc.sides
            grad_lost = ctx.grad_lost

            lost_top = grad_lost.top if not sides.top else 0
            lost_bottom = grad_lost.bottom if not sides.bottom else 0
            lost_left = grad_lost.left if not sides.left else 0
            lost_right = grad_lost.right if not sides.right else 0

            valid_grad = grad_output[
                :,
                :,
                lost_top : grad_output.shape[H_DIM] - lost_bottom,
                lost_left : grad_output.shape[W_DIM] - lost_right,
            ]

            output_stride = ctx.output_stride
            stride_y = float(output_stride[1].item()) if isinstance(output_stride, torch.Tensor) else float(output_stride[1])
            stride_x = float(output_stride[2].item()) if isinstance(output_stride, torch.Tensor) else float(output_stride[2])

            # Keep coordinate mapping consistent with StreamingConv2d and avoid
            # float floor drift on large coordinates.
            data_loc_y = int(round(float(ctx.input_loc.y) / float(stride_y))) + lost_top
            data_loc_x = int(round(float(ctx.input_loc.x) / float(stride_x))) + lost_left
            data_loc = Box(data_loc_y, 0, data_loc_x, 0, ctx.input_loc.sides)

            new_output_box, updated_total_indices = _new_value_indices(valid_grad.shape, data_loc, ctx.seen_indices)

            ctx.seen_indices.y = updated_total_indices.y
            ctx.seen_indices.height = updated_total_indices.height
            ctx.seen_indices.x = updated_total_indices.x
            ctx.seen_indices.width = updated_total_indices.width
            ctx.seen_indices.sides = updated_total_indices.sides

            # We keep tracking seen indices/lost borders for statistics, but
            # propagate grad_input from the full grad_output to mirror
            # StreamingConv2d behavior (which does not mask grad_input).
            del new_output_box

            # interpolate backward is not equivalent to interpolate downsample,
            # so compute grad_input through autograd on interpolate directly.
            with torch.enable_grad():
                proxy_input = inpt.detach().requires_grad_(True)
                proxy_output = torch.nn.functional.interpolate(
                    proxy_input,
                    size=ctx.size,
                    scale_factor=ctx.scale_factor,
                    mode=ctx.mode,
                    align_corners=ctx.align_corners,
                )
                grad_in = torch.autograd.grad(
                    outputs=proxy_output,
                    inputs=proxy_input,
                    grad_outputs=grad_output,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

        return (grad_in, None, None, None, None, None, None, None, None)


upsample = StreamingUpsampleF.apply  # type:ignore


class StreamingUpsample(torch.nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="bilinear", align_corners=False):
        super().__init__()
        if mode != "bilinear":
            raise ValueError("StreamingUpsample only supports bilinear mode.")
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.grad_lost = Lost(0, 0, 0, 0)
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.input_loc = Box(0, 0, 0, 0, None)
        self.output_stride = torch.tensor([1.0, 1.0, 1.0])

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)

    def forward(self, input):
        return upsample(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            self.grad_lost,
            self.seen_indices,
            self.output_stride,
            self.input_loc,
        )
