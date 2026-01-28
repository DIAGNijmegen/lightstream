import torch

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.amp import custom_fwd, custom_bwd

from lightstream.core.scnn.utils import _ntuple, Box, Lost, _new_value_indices, B_DIM, C_DIM, H_DIM, W_DIM


_triple = _ntuple(3)

class StreamingConv2dF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(
        ctx, inpt, weight, bias, stride, padding, dilation, groups, grad_lost, seen_indices, output_stride, input_loc
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
        return torch.nn.functional.conv2d(inpt, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        inpt, weight, bias = ctx.saved_tensors
        grad = grad_weight = grad_bias = None

        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        sides = ctx.input_loc.sides  # Type: Sides
        seen_indices = ctx.seen_indices
        grad_lost = ctx.grad_lost  # Type: Lost
        output_stride = ctx.output_stride
        grad_bias = None
        kernel_size = weight.shape[-1]

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

        grad = grad_output

        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0

        valid_grad = grad[:, :, lost_top : grad.shape[H_DIM] - lost_bottom, lost_left : grad.shape[W_DIM] - lost_right]

        stride, kernel_size, padding = (_triple(stride), _triple(kernel_size), _triple(padding))

        output_stride = output_stride * torch.tensor(stride)
        input_loc = ctx.input_loc

        # Move the location according to how many pixels have been trimmed
        # this will be the location of the valid gradient of this layer in relation
        # to the actual gradient in a normal backpass
        data_loc_y = int(input_loc.y // output_stride[1]) + lost_top
        data_loc_x = int(input_loc.x // output_stride[2]) + lost_left

        data_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)

        # Calculate which part of the gradient is 'new'
        old_value_indices = seen_indices
        new_output_box, updated_total_indices = _new_value_indices(
            valid_grad.shape, data_loc, old_value_indices
        )

        # Update inplace
        seen_indices.y = updated_total_indices.y
        seen_indices.height = updated_total_indices.height
        seen_indices.x = updated_total_indices.x
        seen_indices.width = updated_total_indices.width
        seen_indices.sides = updated_total_indices.sides

        if new_output_box.height > 0 and new_output_box.width > 0:
            relevant_grad = valid_grad[
                :,
                :,
                new_output_box.y : new_output_box.y + new_output_box.height,
                new_output_box.x : new_output_box.x + new_output_box.width,
            ]

            input_y = (new_output_box.y + lost_top) * stride[1]
            input_x = (new_output_box.x + lost_left) * stride[2]

            # Accounting for padding:
            # the kernel locations are relative to the padded input, inpt[0] is not padded
            # this means that the corresponding input of the grad_loc is modules.padding shifted to the left
            # we account for this:
            input_y -= padding[1]
            input_x -= padding[2]
            input_x = max(0, input_x)
            input_y = max(0, input_y)

            relevant_input_height = relevant_grad.shape[H_DIM] * stride[1] + (kernel_size[1] - 1)
            relevant_input_width = relevant_grad.shape[W_DIM] * stride[2] + (kernel_size[2] - 1)
            relevant_input = inpt[
                :, :, input_y : input_y + relevant_input_height, input_x : input_x + relevant_input_width
            ]

            # If layer has padding we need to pad based on if the current tile
            # is at the sides of the input.
            if (padding[0] > 0 or padding[1] > 0 or padding[2] > 0) and (
                sides.top or sides.left or sides.right or sides.bottom
            ):
                # The size of the tile should remain equal.
                crop_bottom = padding[1] if sides.top else 0
                crop_right = padding[2] if sides.left else 0
                relevant_input = inpt[
                    :,
                    :,
                    input_y : input_y + relevant_input_height - crop_bottom,
                    input_x : input_x + relevant_input_width - crop_right,
                ]

                relevant_input = torch.nn.functional.pad(
                    relevant_input,
                    [
                        padding[2] if sides.left else 0,
                        padding[2] if sides.right else 0,
                        padding[1] if sides.top else 0,
                        padding[1] if sides.bottom else 0,
                    ],
                )

            # Calculate the kernel gradients with the new unseen gradient values
            relevant_grad = relevant_grad.contiguous()

            grad_weight = conv2d_weight(
                relevant_input.to(weight.dtype),
                weight.shape,
                relevant_grad.to(weight.dtype),
                stride[1:3],
                (0, 0),  # padding
                dilation,
                groups,
            )

            if bias is not None:
                grad_bias = relevant_grad[0].sum((1, 2))

            del relevant_input
            del relevant_grad
        else:
            # if self.verbose and not hasattr(self, '_inefficient_tile_shape_warning'):
            # print("Warning: no new gradient values found. Tile size could be too small.")
            # self._inefficient_tile_shape_warning = True
            grad_weight = torch.zeros_like(weight)
            if bias is None:
                grad_bias = None
            else:
                grad_bias = torch.zeros_like(bias)

        if bias is not None:
            return (grad_in, grad_weight, grad_bias, None, None, None, None, None, None, None, None)
        else:
            return (grad_in, grad_weight, None, None, None, None, None, None, None, None, None)


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