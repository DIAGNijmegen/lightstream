"""NCHW adapters for NATTEN's native NHWC neighborhood attention layer.

The adapters intentionally do not reimplement neighborhood attention.  NATTEN
remains the sole numerical backend; layout conversion is kept at this boundary
so the rest of Lightstream can consistently use NCHW tensors.
"""

from __future__ import annotations

from importlib import import_module

import torch
from torch import nn
from torch.amp import custom_fwd

from lightstream.core.scnn.utils import Box, Lost


def _pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"expected a scalar or pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _make_natten_attention(**kwargs):
    """Construct the installed NATTEN operator without importing it eagerly."""
    try:
        cls = import_module("natten").NeighborhoodAttention2D
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Neighborhood attention requires the optional NATTEN dependency; "
            "install it with `pip install 'lightstream[nat]'`."
        ) from error
    return cls(**kwargs)


class NeighborhoodAttention2D(nn.Module):
    """NCHW-facing reference wrapper around ``natten.NeighborhoodAttention2D``.

    ``attention`` is useful when adapting an already constructed NAT model.  If
    omitted, all keyword arguments are passed unchanged to the installed NATTEN
    class.  No attention math is performed by this wrapper.
    """

    def __init__(self, dim=None, *, attention=None, **kwargs):
        super().__init__()
        if attention is not None and (dim is not None or kwargs):
            raise ValueError("pass either `attention` or NATTEN constructor arguments, not both")
        if attention is None:
            if dim is None:
                raise TypeError("`dim` is required when `attention` is not supplied")
            attention = _make_natten_attention(dim=dim, **kwargs)
        self.attention = attention
        self._set_spatial_metadata()

    def _set_spatial_metadata(self):
        kernel = _pair(getattr(self.attention, "kernel_size", 3))
        dilation = _pair(getattr(self.attention, "dilation", 1) or 1)
        if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
            raise ValueError("streaming neighborhood attention requires odd kernel sizes")
        self.kernel_size = kernel
        self.dilation = dilation
        self.stride = (1, 1)
        radius_h = dilation[0] * (kernel[0] - 1) // 2
        radius_w = dilation[1] * (kernel[1] - 1) // 2
        self.directional_spatial_support = Lost(radius_h, radius_w, radius_h, radius_w)
        self.padding = (radius_h, radius_w)
        # A concise alias for consumers which do not need to distinguish the
        # forward support from other module metadata.
        self.spatial_support = self.directional_spatial_support

    @classmethod
    def from_natten(cls, module: nn.Module) -> "NeighborhoodAttention2D":
        return cls(attention=module)

    def to_natten(self) -> nn.Module:
        return self.attention

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 4:
            raise ValueError(f"expected an NCHW rank-4 tensor, got shape {tuple(input.shape)}")
        output = self.attention(input.permute(0, 2, 3, 1).contiguous())
        return output.permute(0, 3, 1, 2).contiguous()


class StreamingNeighborhoodAttention2D(NeighborhoodAttention2D):
    """Streaming form of :class:`NeighborhoodAttention2D`.

    Lightstream supplies overlapping halo tiles and retains unique valid query
    regions.  NATTEN still computes both the forward and backward math.  The
    small autograd boundary below only de-duplicates global queries when an
    uneven final tile is shifted back and overlaps its predecessor; gradients
    from retained queries continue to flow through every key and value.
    """

    def __init__(self, dim=None, *, attention=None, **kwargs):
        super().__init__(dim, attention=attention, **kwargs)
        self.grad_lost = self.directional_spatial_support
        self.output_stride = torch.tensor([1, 1, 1], dtype=torch.long)
        self.reset()

    @classmethod
    def from_reference(cls, module: NeighborhoodAttention2D):
        converted = cls(attention=module.attention)
        converted.train(module.training)
        return converted

    def to_reference(self) -> NeighborhoodAttention2D:
        converted = NeighborhoodAttention2D(attention=self.attention)
        converted.train(self.training)
        return converted

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.input_loc = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.input_loc is None or not torch.is_grad_enabled():
            return super().forward(input)
        parameters = tuple(self.attention.parameters())
        return _StreamingNeighborhoodAttentionFunction.apply(
            input,
            self.attention,
            self.seen_indices,
            self.input_loc,
            self.grad_lost,
            self.output_stride,
            *parameters,
        )


class _StreamingNeighborhoodAttentionFunction(torch.autograd.Function):
    """Recompute NATTEN after output ownership has already been assigned.

    ``StreamingCNN`` selects the globally owned output queries before invoking
    autograd for a replay tile.  Consequently ``grad_output`` is already the
    exact gradient which must reach the attention operand of a residual merge.
    Applying the attention module's own rectangular ``seen`` filter here would
    assign ownership a second time.  In particular, that loses gradients after
    two residual branches have first been summed at a shared tensor.

    The recomputation deliberately returns its complete input gradient.  Values
    in a neighbouring tile's *output* ownership region can be key/value halo
    dependencies of this tile's owned queries and therefore must accumulate in
    the global input rather than being de-duplicated.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, attention, seen, input_loc, grad_lost, output_stride, *parameters):
        ctx.attention = attention
        ctx.seen = seen
        ctx.input_loc = input_loc
        ctx.grad_lost = grad_lost
        ctx.output_stride = output_stride
        ctx.save_for_backward(input, *parameters)
        with torch.no_grad():
            output = attention(input.permute(0, 2, 3, 1).contiguous())
            return output.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        input, *parameters = ctx.saved_tensors
        with torch.enable_grad():
            replay_dtype = next(
                parameter.dtype
                for parameter in parameters
                if parameter.is_floating_point()
            )
            replay_input = input.detach().to(dtype=replay_dtype).requires_grad_(True)
            replay = ctx.attention(
                replay_input.permute(0, 2, 3, 1).contiguous()
            )
            replay = replay.permute(0, 3, 1, 2).contiguous()
            trainable_indices = [index for index, parameter in enumerate(parameters) if parameter.requires_grad]
            input_gradient = torch.autograd.grad(
                replay,
                replay_input,
                grad_output,
                retain_graph=bool(trainable_indices),
            )[0]
            parameter_gradients = ()
            if trainable_indices:
                parameter_gradients = torch.autograd.grad(
                    replay,
                    tuple(parameters[index] for index in trainable_indices),
                    grad_output,
                    allow_unused=True,
                )
        input_gradient = input_gradient.to(dtype=input.dtype)
        parameter_grads = [None] * len(parameters)
        for index, gradient in zip(trainable_indices, parameter_gradients):
            parameter_grads[index] = gradient
        return (input_gradient, None, None, None, None, None, *parameter_grads)


# Explicit aliases make the layout contract discoverable and retain a short
# spelling for model conversion code.
NCHWNeighborhoodAttention2D = NeighborhoodAttention2D
StreamingNeighborhoodAttention = StreamingNeighborhoodAttention2D


__all__ = [
    "NCHWNeighborhoodAttention2D",
    "NeighborhoodAttention2D",
    "StreamingNeighborhoodAttention",
    "StreamingNeighborhoodAttention2D",
]
