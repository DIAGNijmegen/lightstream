from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, Lost, _new_value_indices, H_DIM, W_DIM


class ChannelLayerNorm(nn.Module):
    """Apply layer normalization across channels for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ChannelLayerNorm expects 4D NCHW input, got {x.ndim}D input with shape {tuple(x.shape)}.")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"ChannelLayerNorm expected {self.num_channels} channels at dimension 1, "
                f"got {x.shape[1]} channels for input shape {tuple(x.shape)}."
            )

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class StreamingChannelLayerNormF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(ctx, inpt, weight, bias, num_channels, eps, grad_lost, seen_indices, output_stride, input_loc):
        ctx.num_channels = num_channels
        ctx.eps = eps
        ctx.has_weight = weight is not None
        ctx.has_bias = bias is not None
        ctx.grad_lost = grad_lost
        ctx.seen_indices = seen_indices
        ctx.output_stride = output_stride
        ctx.input_loc = input_loc

        tensors = (inpt, weight) if weight is not None else (inpt,)
        ctx.save_for_backward(*tensors)

        out = F.layer_norm(inpt.permute(0, 2, 3, 1), (num_channels,), weight, bias, eps)
        return out.permute(0, 3, 1, 2)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        inpt = saved[0]
        weight = saved[1] if ctx.has_weight else None

        centered = inpt - inpt.mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(centered.pow(2).mean(dim=1, keepdim=True) + ctx.eps)
        x_hat = centered * inv_std

        grad_in = None
        if ctx.needs_input_grad[0]:
            grad_norm = grad_output
            if weight is not None:
                grad_norm = grad_norm * weight.to(dtype=grad_output.dtype).view(1, -1, 1, 1)

            channels = inpt.shape[1]
            grad_mean = grad_norm.mean(dim=1, keepdim=True)
            grad_xhat_mean = (grad_norm * x_hat).mean(dim=1, keepdim=True)
            grad_in = (grad_norm - grad_mean - x_hat * grad_xhat_mean) * inv_std

        grad_weight = grad_bias = None
        if ctx.has_weight or ctx.has_bias:
            input_loc = ctx.input_loc
            sides = input_loc.sides if input_loc is not None else None
            grad_lost = ctx.grad_lost
            seen_indices = ctx.seen_indices

            lost_top = grad_lost.top if not (sides is not None and sides.top) else 0
            lost_bottom = grad_lost.bottom if not (sides is not None and sides.bottom) else 0
            lost_left = grad_lost.left if not (sides is not None and sides.left) else 0
            lost_right = grad_lost.right if not (sides is not None and sides.right) else 0

            valid_grad = grad_output[
                :,
                :,
                lost_top : grad_output.shape[H_DIM] - lost_bottom,
                lost_left : grad_output.shape[W_DIM] - lost_right,
            ]

            if input_loc is None:
                data_loc = Box(lost_top, 0, lost_left, 0, sides)
            else:
                output_stride = ctx.output_stride
                stride_h = int(output_stride[1].item()) if isinstance(output_stride, torch.Tensor) else int(output_stride[1])
                stride_w = int(output_stride[2].item()) if isinstance(output_stride, torch.Tensor) else int(output_stride[2])
                data_loc_y = int(input_loc.y // stride_h) + lost_top
                data_loc_x = int(input_loc.x // stride_w) + lost_left
                data_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)

            new_output_box, updated_total_indices = _new_value_indices(valid_grad.shape, data_loc, seen_indices)

            seen_indices.y = updated_total_indices.y
            seen_indices.height = updated_total_indices.height
            seen_indices.x = updated_total_indices.x
            seen_indices.width = updated_total_indices.width
            seen_indices.sides = updated_total_indices.sides

            if new_output_box.height > 0 and new_output_box.width > 0:
                y0 = lost_top + new_output_box.y
                y1 = y0 + new_output_box.height
                x0 = lost_left + new_output_box.x
                x1 = x0 + new_output_box.width
                relevant_grad = grad_output[:, :, y0:y1, x0:x1]
                if ctx.has_weight:
                    relevant_x_hat = x_hat[:, :, y0:y1, x0:x1]
                    grad_weight = (relevant_grad * relevant_x_hat).sum(dim=(0, 2, 3)).to(dtype=weight.dtype)
                if ctx.has_bias:
                    grad_bias = relevant_grad.sum(dim=(0, 2, 3)).to(dtype=weight.dtype)
            else:
                if ctx.has_weight:
                    grad_weight = torch.zeros_like(weight)
                if ctx.has_bias:
                    grad_bias = torch.zeros_like(weight)

        return grad_in, grad_weight, grad_bias, None, None, None, None, None, None


channel_layer_norm = StreamingChannelLayerNormF.apply


class StreamingChannelLayerNorm(nn.Module):
    """Streaming channel layer normalization for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.grad_lost = Lost(0, 0, 0, 0)
        self.output_stride = torch.tensor([1, 1, 1])
        self.reset()

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.input_loc = Box(0, 0, 0, 0, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"StreamingChannelLayerNorm expects 4D NCHW input, got {x.ndim}D input with shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"StreamingChannelLayerNorm expected {self.num_channels} channels at dimension 1, "
                f"got {x.shape[1]} channels for input shape {tuple(x.shape)}."
            )

        return channel_layer_norm(
            x,
            self.weight,
            self.bias,
            self.num_channels,
            self.eps,
            self.grad_lost,
            self.seen_indices,
            self.output_stride,
            self.input_loc,
        )

    @classmethod
    def from_channel_layer_norm(cls, module: ChannelLayerNorm) -> "StreamingChannelLayerNorm":
        mod = cls(module.num_channels, module.norm.eps, module.norm.elementwise_affine)
        if module.norm.elementwise_affine:
            mod = mod.to(module.norm.weight.device, non_blocking=True)
            mod = mod.to(module.norm.weight.dtype)
            mod.weight.data.copy_(module.norm.weight.data)
            mod.bias.data.copy_(module.norm.bias.data)
            mod.weight.requires_grad = module.norm.weight.requires_grad
            mod.bias.requires_grad = module.norm.bias.requires_grad
        return mod

    def to_channel_layer_norm(self) -> ChannelLayerNorm:
        mod = ChannelLayerNorm(self.num_channels, self.eps, self.elementwise_affine)
        if self.elementwise_affine:
            mod = mod.to(self.weight.device, non_blocking=True)
            mod = mod.to(self.weight.dtype)
            mod.norm.weight.data.copy_(self.weight.data)
            mod.norm.bias.data.copy_(self.bias.data)
            mod.norm.weight.requires_grad = self.weight.requires_grad
            mod.norm.bias.requires_grad = self.bias.requires_grad
        return mod
