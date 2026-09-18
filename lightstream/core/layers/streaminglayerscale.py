from __future__ import annotations

from collections.abc import Iterable
import torch
import torch.nn as nn
from torch.amp import custom_bwd, custom_fwd

from lightstream.core.scnn.utils import Box, H_DIM, Lost, W_DIM, _new_value_indices


def _normalize_weight_shape(shape: int | Iterable[int] | torch.Size) -> torch.Size:
    if isinstance(shape, torch.Size):
        values = tuple(shape)
    elif isinstance(shape, int):
        values = (shape,)
    elif isinstance(shape, Iterable):
        values = tuple(shape)
    else:
        raise TypeError(
            "LayerScale shape must be an int, tuple/list of ints, or torch.Size; "
            f"got {type(shape).__name__}."
        )

    if not all(isinstance(dim, int) for dim in values):
        raise TypeError(f"LayerScale shape must contain only ints, got {values!r}.")
    if any(dim < 0 for dim in values):
        raise ValueError(
            f"LayerScale shape dimensions must be non-negative, got {values!r}."
        )
    return torch.Size(values)


def _broadcast_error(
    module_name: str, x: torch.Tensor, scale: torch.Tensor
) -> ValueError:
    return ValueError(
        f"{module_name} scale with shape {tuple(scale.shape)} cannot broadcast to input shape {tuple(x.shape)}. "
        "Choose a scale shape compatible with PyTorch broadcasting, for example (1,), (C,), or (1, C, 1, 1)."
    )


class LayerScale(nn.Module):
    """Learnable multiplicative scale with PyTorch broadcasting semantics."""

    def __init__(
        self, shape: int | Iterable[int] | torch.Size, init_value: float = 0.0
    ):
        super().__init__()
        self.shape = _normalize_weight_shape(shape)
        self.init_value = init_value
        self.weight = nn.Parameter(torch.full(self.shape, init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return x * self.weight
        except RuntimeError as error:
            raise _broadcast_error(type(self).__name__, x, self.weight) from error


class StreamingLayerScaleF(torch.autograd.Function):
    @staticmethod
    @custom_fwd(
        device_type="cuda",
        cast_inputs=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    def forward(ctx, inpt, weight, grad_lost, seen_indices, output_stride, input_loc):
        ctx.grad_lost = grad_lost
        ctx.seen_indices = seen_indices
        ctx.output_stride = output_stride
        ctx.input_loc = input_loc
        ctx.save_for_backward(inpt, weight)
        return inpt * weight

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        inpt, weight = ctx.saved_tensors

        grad_in = (
            grad_output * weight.to(dtype=grad_output.dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_weight = None

        if ctx.needs_input_grad[1]:
            if grad_output.ndim < 4:
                grad_weight = (
                    (grad_output * inpt).sum_to_size(weight.shape).to(dtype=weight.dtype)
                )
                return grad_in, grad_weight, None, None, None, None

            input_loc = ctx.input_loc
            sides = input_loc.sides if input_loc is not None else None
            grad_lost = ctx.grad_lost
            seen_indices = ctx.seen_indices

            lost_top = grad_lost.top if not (sides is not None and sides.top) else 0
            lost_bottom = (
                grad_lost.bottom if not (sides is not None and sides.bottom) else 0
            )
            lost_left = grad_lost.left if not (sides is not None and sides.left) else 0
            lost_right = (
                grad_lost.right if not (sides is not None and sides.right) else 0
            )

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
                stride_h = (
                    int(output_stride[1].item())
                    if isinstance(output_stride, torch.Tensor)
                    else int(output_stride[1])
                )
                stride_w = (
                    int(output_stride[2].item())
                    if isinstance(output_stride, torch.Tensor)
                    else int(output_stride[2])
                )
                data_loc_y = int(input_loc.y // stride_h) + lost_top
                data_loc_x = int(input_loc.x // stride_w) + lost_left
                data_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)

            new_output_box, updated_total_indices = _new_value_indices(
                valid_grad.shape, data_loc, seen_indices
            )
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
                relevant_input = inpt[:, :, y0:y1, x0:x1]
                grad_weight = (
                    (relevant_grad * relevant_input)
                    .sum_to_size(weight.shape)
                    .to(dtype=weight.dtype)
                )
            else:
                grad_weight = torch.zeros_like(weight)

        return grad_in, grad_weight, None, None, None, None


streaming_layer_scale = StreamingLayerScaleF.apply


class StreamingLayerScale(nn.Module):
    """Streaming variant of :class:`LayerScale` with compatible ``scale`` state dict keys."""

    def __init__(
        self, shape: int | Iterable[int] | torch.Size, init_value: float = 0.0
    ):
        super().__init__()
        self.shape = _normalize_weight_shape(shape)
        self.init_value = init_value
        self.weight = nn.Parameter(torch.full(self.shape, init_value))
        self.grad_lost = Lost(0, 0, 0, 0)
        self.output_stride = torch.tensor([1, 1, 1])
        self.reset()

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.input_loc = Box(0, 0, 0, 0, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return streaming_layer_scale(
                x,
                self.weight,
                self.grad_lost,
                self.seen_indices,
                self.output_stride,
                self.input_loc,
            )
        except RuntimeError as error:
            raise _broadcast_error(type(self).__name__, x, self.scale) from error

    @classmethod
    def from_layer_scale(cls, module: LayerScale) -> "StreamingLayerScale":
        if not isinstance(module, LayerScale):
            raise TypeError(
                f"StreamingLayerScale.from_layer_scale expected LayerScale, got {type(module).__name__}."
            )
        mod = cls(module.weight.shape, module.init_value)
        mod = mod.to(
            device=module.weight.device, dtype=module.weight.dtype, non_blocking=True
        )
        mod.weight.requires_grad = module.weight.requires_grad
        mod.weight.data.copy_(module.weight.data)
        return mod

    def to_layer_scale(self) -> LayerScale:
        mod = LayerScale(self.weight.shape, self.init_value)
        mod = mod.to(
            device=self.weight.device, dtype=self.weight.dtype, non_blocking=True
        )
        mod.weight.requires_grad = self.weight.requires_grad
        mod.weight.data.copy_(self.weight.data)
        return mod
