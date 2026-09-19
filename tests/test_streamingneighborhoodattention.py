import copy

import torch
from torch import nn

from lightstream.core.layers.streamingneighborhoodattention import (
    NeighborhoodAttention2D,
    StreamingNeighborhoodAttention2D,
)
from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.core.scnn.utils import Lost


class _LocalNHWCBackend(nn.Module):
    """Small differentiable stand-in used to test the adapter without NATTEN."""

    def __init__(self, channels=4, kernel_size=3, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = nn.Parameter(torch.linspace(0.7, 1.3, channels))

    def forward(self, value):
        radius = self.dilation * (self.kernel_size - 1) // 2
        nchw = value.permute(0, 3, 1, 2)
        weight = torch.ones(
            nchw.shape[1], 1, self.kernel_size, self.kernel_size, device=nchw.device, dtype=nchw.dtype
        ) / self.kernel_size**2
        local = torch.nn.functional.conv2d(
            nchw,
            weight,
            stride=1,
            padding=radius,
            dilation=self.dilation,
            groups=nchw.shape[1],
        )
        return (local * self.scale[None, :, None, None]).permute(0, 2, 3, 1)


def _manual_halo_tiles(module, image, query_shape):
    """Run clipped halo tiles and retain every global query exactly once."""
    support = module.directional_spatial_support
    rows = []
    for query_y in range(0, image.shape[-2], query_shape[0]):
        query_bottom = min(query_y + query_shape[0], image.shape[-2])
        columns = []
        for query_x in range(0, image.shape[-1], query_shape[1]):
            query_right = min(query_x + query_shape[1], image.shape[-1])
            halo_y = max(0, query_y - support.top)
            halo_x = max(0, query_x - support.left)
            halo_bottom = min(image.shape[-2], query_bottom + support.bottom)
            halo_right = min(image.shape[-1], query_right + support.right)
            tile = module(image[:, :, halo_y:halo_bottom, halo_x:halo_right])
            columns.append(
                tile[
                    :,
                    :,
                    query_y - halo_y : query_bottom - halo_y,
                    query_x - halo_x : query_right - halo_x,
                ]
            )
        rows.append(torch.cat(columns, dim=-1))
    return torch.cat(rows, dim=-2)


def test_nchw_wrapper_and_manual_halo_oracle_match_values_gradients_and_step():
    torch.manual_seed(5)
    reference = NeighborhoodAttention2D(attention=_LocalNHWCBackend(kernel_size=5, dilation=2))
    tiled = copy.deepcopy(reference)
    reference_image = torch.randn(2, 4, 17, 19, requires_grad=True)
    tiled_image = reference_image.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(reference_image)

    reference_output = reference(reference_image)
    tiled_output = _manual_halo_tiles(tiled, tiled_image, query_shape=(6, 7))
    torch.testing.assert_close(tiled_output, reference_output)

    reference_output.backward(upstream)
    tiled_output.backward(upstream)
    torch.testing.assert_close(tiled_image.grad, reference_image.grad)
    reference_grads = dict(reference.named_parameters())
    tiled_grads = dict(tiled.named_parameters())
    assert reference_grads.keys() == tiled_grads.keys()
    for name in reference_grads:
        torch.testing.assert_close(tiled_grads[name].grad, reference_grads[name].grad)

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    tiled_optimizer = torch.optim.SGD(tiled.parameters(), lr=0.03)
    reference_optimizer.step()
    tiled_optimizer.step()
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(dict(tiled.named_parameters())[name], parameter)


def test_streaming_metadata_conversion_cache_and_restoration():
    module = NeighborhoodAttention2D(attention=_LocalNHWCBackend(kernel_size=3))
    reference_module = copy.deepcopy(module)
    image = torch.randn(2, 4, 11, 13, requires_grad=True)
    streaming_image = image.detach().clone().requires_grad_(True)
    reference = reference_module(image)
    streaming = StreamingCNN(module, tile_shape=(2, 4, 7, 8), copy_to_gpu=True)

    assert isinstance(streaming.stream_module, StreamingNeighborhoodAttention2D)
    assert streaming.stream_module.stride == (1, 1)
    assert streaming.stream_module.directional_spatial_support == Lost(1, 1, 1, 1)
    torch.testing.assert_close(streaming(streaming_image.detach()), reference)
    cache = streaming.get_tile_cache()
    assert cache["net_stats"][""]["directional_spatial_support"] == Lost(1, 1, 1, 1)

    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    streaming.backward(streaming_image, upstream)
    torch.testing.assert_close(streaming_image.grad, image.grad)
    reference_parameters = dict(reference_module.named_parameters())
    streaming_parameters = dict(streaming.stream_module.named_parameters())
    assert reference_parameters.keys() == streaming_parameters.keys()
    for name, parameter in reference_parameters.items():
        torch.testing.assert_close(streaming_parameters[name].grad, parameter.grad)

    streaming.disable()
    assert type(streaming.stream_module) is NeighborhoodAttention2D
