import torch
import torch.nn as nn

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.amp import custom_fwd, custom_bwd

from lightstream.core.scnn.utils import (
    _new_value_indices,
    Box,
    Lost,
)


class StreamingConv2dF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(
        device_type="cuda",
        cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    def forward(
        ctx,
        inpt,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        grad_lost,
        seen_indices,
        output_stride,
        input_loc,
    ):
        ctx.save_for_backward(inpt, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.grad_lost = grad_lost
        ctx.seen_indices = seen_indices
        ctx.output_stride = output_stride
        ctx.input_loc = input_loc
        return torch.nn.functional.conv2d(
            inpt, weight, bias, stride, padding, dilation, groups
        )

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        inpt, weight, bias = ctx.saved_tensors
        grad = grad_weight = grad_bias = None

        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_bias = None

        if ctx.needs_input_grad[0]:
            # TODO: performance improvements possible by only backpropping valid input
            # grad_input_padding = _grad_input_padding(grad_output, inpt.shape, stride, padding, (weight.shape[2], weight.shape[3]))
            # TODO: use this!?
            grad_in = torch.nn.grad.conv2d_input(
                inpt.shape,
                weight.to(inpt.dtype),
                grad_output,
                stride,  # type:ignore
                padding,
                dilation,
                groups,
            )
        else:
            grad_in = None

        # The replay head already assigns every output query to exactly one
        # tile.  Gradients at an internal activation coordinate can therefore
        # occur in several tiles only because distinct owned queries depend on
        # that activation.  Those are additive contributions, not duplicates.
        # Computing each tile's complete convolution gradient lets autograd's
        # normal parameter and input-view accumulation assemble the global
        # gradient at the correct boundary.
        grad_weight = torch.nn.grad.conv2d_weight(
            inpt.to(weight.dtype),
            weight.shape,
            grad_output.to(weight.dtype).contiguous(),
            stride,
            padding,
            dilation,
            groups,
        )
        if bias is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3)).to(bias.dtype)

        # Preserve the traversal cursor for saliency stitching and diagnostics,
        # but do not use it to filter dependency-gradient contributions.
        sides = ctx.input_loc.sides
        lost_top = ctx.grad_lost.top if not sides.top else 0
        lost_bottom = ctx.grad_lost.bottom if not sides.bottom else 0
        lost_left = ctx.grad_lost.left if not sides.left else 0
        lost_right = ctx.grad_lost.right if not sides.right else 0
        valid_grad = grad_output[
            :,
            :,
            lost_top : grad_output.shape[2] - lost_bottom,
            lost_left : grad_output.shape[3] - lost_right,
        ]
        output_stride = ctx.output_stride * torch.tensor((1, *ctx.stride))
        data_loc = Box(
            int(ctx.input_loc.y // output_stride[1]) + lost_top,
            0,
            int(ctx.input_loc.x // output_stride[2]) + lost_left,
            0,
            sides,
        )
        _, updated = _new_value_indices(valid_grad.shape, data_loc, ctx.seen_indices)
        ctx.seen_indices.y = updated.y
        ctx.seen_indices.height = updated.height
        ctx.seen_indices.x = updated.x
        ctx.seen_indices.width = updated.width
        ctx.seen_indices.sides = updated.sides

        if bias is not None:
            return (
                grad_in,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            return (
                grad_in,
                grad_weight,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


conv2d = StreamingConv2dF.apply  # type:ignore


class StreamingConv2d(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(StreamingConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )
        self.grad_lost = Lost(0, 0, 0, 0)
        self.reset()

    @classmethod
    def from_torch_conv2d(cls, module: nn.Conv2d) -> "StreamingConv2d":
        mod = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
        )
        mod = mod.to(module.weight.device, non_blocking=True)
        mod = mod.to(module.weight.dtype)
        mod.load_state_dict(module.state_dict())
        mod.weight.requires_grad = module.weight.requires_grad
        if module.bias is not None:
            mod.bias.requires_grad = module.bias.requires_grad
        return mod

    def to_torch_conv2d(self) -> nn.Conv2d:
        mod = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
        )
        mod = mod.to(self.weight.device, non_blocking=True)
        mod = mod.to(self.weight.dtype)
        mod.load_state_dict(self.state_dict())
        mod.weight.requires_grad = self.weight.requires_grad
        if self.bias is not None:
            mod.bias.requires_grad = self.bias.requires_grad
        return mod

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.input_loc = Box(0, 0, 0, 0, None)
        self.tile_output_box = Box(0, 0, 0, 0, None)

    def forward(self, input):
        return conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.grad_lost,
            self.seen_indices,
            self.output_stride,
            self.input_loc,
        )
