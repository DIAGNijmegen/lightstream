import copy

import pytest
import torch
from torch import nn

from lightstream.core.layers.streamingneighborhoodattention import (
    NeighborhoodAttention2D,
    StreamingNeighborhoodAttention2D,
)
from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.core.scnn.utils import Lost


@pytest.fixture(scope="session")
def natten_backend():
    """Load the real optional backend only for tests which require it."""
    return pytest.importorskip("natten")


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
    def _enlarge_axis(start, end, query_start, query_end, image_extent, minimum_extent):
        """Expand an interval to the backend's minimum supported axis extent."""
        if minimum_extent > image_extent:
            raise ValueError(
                f"minimum tile extent {minimum_extent} exceeds image extent {image_extent}"
            )

        missing = max(0, minimum_extent - (end - start))
        if start == 0:
            end += missing
        elif end == image_extent:
            start -= missing
        else:
            grow_end = min(missing, image_extent - end)
            end += grow_end
            start -= missing - grow_end

        assert 0 <= start <= query_start <= query_end <= end <= image_extent
        assert end - start >= minimum_extent
        return start, end

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
            halo_y, halo_bottom = _enlarge_axis(
                halo_y,
                halo_bottom,
                query_y,
                query_bottom,
                image.shape[-2],
                module.kernel_size[0] * module.dilation[0],
            )
            halo_x, halo_right = _enlarge_axis(
                halo_x,
                halo_right,
                query_x,
                query_right,
                image.shape[-1],
                module.kernel_size[1] * module.dilation[1],
            )
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


def _assert_named_parameter_state_matches(actual, expected, attribute=None):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name in expected_parameters:
        actual_value = actual_parameters[name]
        expected_value = expected_parameters[name]
        if attribute is not None:
            actual_value = getattr(actual_value, attribute)
            expected_value = getattr(expected_value, attribute)
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2e-4,
            atol=2e-5,
            msg=f"named parameter {name!r} {attribute or 'value'} differs",
        )


def _region_masks(height, width, query_shape, radius):
    """Return masks that make tile-boundary and image-boundary failures legible."""
    y = torch.arange(height)[:, None]
    x = torch.arange(width)[None, :]
    horizontal = torch.zeros((height, width), dtype=torch.bool)
    vertical = torch.zeros_like(horizontal)
    for seam in range(query_shape[0], height, query_shape[0]):
        horizontal |= (y - seam).abs() <= radius
    for seam in range(query_shape[1], width, query_shape[1]):
        vertical |= (x - seam).abs() <= radius

    true_edge = (y == 0) | (y == height - 1) | (x == 0) | (x == width - 1)
    corner = ((y == 0) | (y == height - 1)) & ((x == 0) | (x == width - 1))
    return {
        "horizontal seams": horizontal,
        "vertical seams": vertical,
        "seam intersections": horizontal & vertical,
        "true edges": true_edge,
        "corners": corner,
    }


def _assert_spatial_regions_match(actual, expected, query_shape, radius, quantity):
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5, msg=f"complete {quantity}")
    for region, mask in _region_masks(*expected.shape[-2:], query_shape, radius).items():
        assert mask.any(), f"test geometry did not create {region}"
        torch.testing.assert_close(
            actual[..., mask],
            expected[..., mask],
            rtol=2e-4,
            atol=2e-5,
            msg=f"{quantity} differs at {region}",
        )


def _make_natten(natten_backend, channels, heads, kernel_size, dilation):
    # Keep both attention-weight and projection dropout disabled so tiled and
    # untiled executions are deterministic and directly comparable.
    attention = natten_backend.NeighborhoodAttention2D(
        dim=channels,
        num_heads=heads,
        kernel_size=kernel_size,
        dilation=dilation,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    assert attention.attn_drop.p == 0.0
    assert attention.proj_drop.p == 0.0
    return attention.float()


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


def test_real_natten_nhwc_matches_untiled_nchw_wrapper(natten_backend):
    """The adapter must be only a layout conversion around NATTEN 0.21.7."""
    torch.manual_seed(101)
    direct = _make_natten(natten_backend, channels=8, heads=2, kernel_size=3, dilation=1)
    wrapped = NeighborhoodAttention2D(attention=copy.deepcopy(direct))
    direct_input = torch.randn(1, 11, 13, 8, dtype=torch.float32, requires_grad=True)
    wrapped_input = direct_input.detach().permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    direct_upstream = torch.randn_like(direct_input)
    wrapped_upstream = direct_upstream.permute(0, 3, 1, 2).contiguous()

    direct_output = direct(direct_input)
    wrapped_output = wrapped(wrapped_input)
    torch.testing.assert_close(
        wrapped_output,
        direct_output.permute(0, 3, 1, 2),
        rtol=2e-4,
        atol=2e-5,
    )

    direct_output.backward(direct_upstream)
    wrapped_output.backward(wrapped_upstream)
    torch.testing.assert_close(
        wrapped_input.grad,
        direct_input.grad.permute(0, 3, 1, 2),
        rtol=2e-4,
        atol=2e-5,
    )
    _assert_named_parameter_state_matches(wrapped.attention, direct, attribute="grad")

    torch.optim.SGD(direct.parameters(), lr=0.025).step()
    torch.optim.SGD(wrapped.parameters(), lr=0.025).step()
    _assert_named_parameter_state_matches(wrapped.attention, direct)


@pytest.mark.parametrize(
    ("kernel_size", "dilation", "heads", "batch", "image_shape", "query_shape"),
    [
        pytest.param(3, 1, 1, 1, (17, 19), (6, 7), id="baseline-k3-d1-h1-b1"),
        pytest.param(3, 2, 2, 2, (19, 22), (7, 8), id="k3-d2-h2-b2"),
        pytest.param(5, 1, 4, 1, (18, 23), (7, 9), id="k5-d1-h4-b1"),
        pytest.param(5, 2, 2, 2, (21, 25), (6, 7), id="k5-d2-h2-b2"),
        pytest.param(7, 1, 2, 2, (20, 24), (7, 10), id="k7-d1-h2-b2"),
        pytest.param(7, 2, 4, 1, (23, 27), (7, 8), id="k7-d2-h4-b1"),
    ],
)
def test_real_natten_full_manual_and_streaming_match_everywhere(
    natten_backend, kernel_size, dilation, heads, batch, image_shape, query_shape
):
    """Compare values, all gradients, and an SGD step across all execution paths."""
    torch.manual_seed(1000 + kernel_size * 10 + dilation + heads + batch)
    channels = 8
    base = NeighborhoodAttention2D(
        attention=_make_natten(natten_backend, channels, heads, kernel_size, dilation)
    )
    full_module = copy.deepcopy(base)
    manual_module = copy.deepcopy(base)
    streaming_source = copy.deepcopy(base)

    shared_input = torch.randn(
        batch, channels, *image_shape, dtype=torch.float32, requires_grad=True
    )
    full_input = shared_input.detach().clone().requires_grad_(True)
    manual_input = shared_input.detach().clone().requires_grad_(True)
    streaming_input = shared_input.detach().clone().requires_grad_(True)
    upstream = torch.randn(batch, channels, *image_shape, dtype=torch.float32)
    radius = dilation * (kernel_size - 1) // 2

    # A physical stream tile consists of its uniquely-owned query area plus
    # both halos.  Non-divisible image/query sizes exercise shifted final tiles.
    tile_hw = (query_shape[0] + 2 * radius, query_shape[1] + 2 * radius)
    assert image_shape[0] % query_shape[0] and image_shape[1] % query_shape[1]
    streaming = StreamingCNN(
        streaming_source,
        tile_shape=(batch, channels, *tile_hw),
        copy_to_gpu=True,
    )

    full_output = full_module(full_input)
    manual_output = _manual_halo_tiles(manual_module, manual_input, query_shape)
    streaming_output = streaming(streaming_input)
    _assert_spatial_regions_match(manual_output, full_output, query_shape, radius, "manual output")
    _assert_spatial_regions_match(
        streaming_output, full_output, query_shape, radius, "StreamingCNN output"
    )

    full_output.backward(upstream)
    manual_output.backward(upstream)
    streaming.backward(streaming_input, upstream)
    _assert_spatial_regions_match(
        manual_input.grad, full_input.grad, query_shape, radius, "manual input gradient"
    )
    _assert_spatial_regions_match(
        streaming_input.grad,
        full_input.grad,
        query_shape,
        radius,
        "StreamingCNN input gradient",
    )
    _assert_named_parameter_state_matches(manual_module, full_module, attribute="grad")
    _assert_named_parameter_state_matches(
        streaming.stream_module, full_module, attribute="grad"
    )

    for module in (full_module, manual_module, streaming.stream_module):
        torch.optim.SGD(module.parameters(), lr=0.025).step()
    _assert_named_parameter_state_matches(manual_module, full_module)
    _assert_named_parameter_state_matches(streaming.stream_module, full_module)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param((3, 1), (3, 1), id="k3-d1-then-k3-d1"),
        pytest.param((7, 1), (7, 1), id="k7-d1-then-k7-d1"),
        pytest.param((3, 1), (3, 2), id="k3-d1-then-k3-d2"),
        pytest.param((7, 1), (7, 2), id="k7-d1-then-k7-d2"),
    ],
)
def test_two_real_natten_layers_match_and_reset_unique_queries(
    natten_backend, first, second
):
    """Two independent attention layers accumulate halos and reset replay state."""
    torch.manual_seed(7000 + first[0] * 100 + second[0] * 10 + second[1])
    batch, channels, heads = 1, 8, 2

    def make_layer(kernel_and_dilation):
        kernel_size, dilation = kernel_and_dilation
        return NeighborhoodAttention2D(
            attention=_make_natten(
                natten_backend,
                channels=channels,
                heads=heads,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )

    # Construct both wrappers separately: sharing a NATTEN module here would
    # conceal parameter-gradient and unique-query bookkeeping bugs.
    base = nn.Sequential(make_layer(first), make_layer(second))
    full_module = copy.deepcopy(base)
    streaming_source = copy.deepcopy(base)
    assert base[0].attention is not base[1].attention

    radii = tuple(
        dilation * (kernel_size - 1) // 2 for kernel_size, dilation in (first, second)
    )
    accumulated_radius = sum(radii)
    expected_lost = Lost(
        accumulated_radius,
        accumulated_radius,
        accumulated_radius,
        accumulated_radius,
    )

    # The physical tile leaves a 5x7 valid interior after both receptive
    # supports have accumulated.  Pick non-divisible full-image dimensions so
    # the last tile is shifted on both axes rather than landing on the grid.
    query_shape = (5, 7)
    tile_hw = tuple(size + 2 * accumulated_radius for size in query_shape)
    image_shape = [tile_hw[0] + query_shape[0] + 1, tile_hw[1] + query_shape[1] + 1]
    for axis, query_extent in enumerate(query_shape):
        while image_shape[axis] % query_extent == 0:
            image_shape[axis] += 1
    image_shape = tuple(image_shape)
    assert all(tile_hw[axis] - 2 * accumulated_radius > 0 for axis in range(2))
    assert all(image_shape[axis] % query_shape[axis] for axis in range(2))

    streaming = StreamingCNN(
        streaming_source,
        tile_shape=(batch, channels, *tile_hw),
        copy_to_gpu=True,
    )
    streamed_layers = list(streaming.stream_module)
    assert all(
        isinstance(layer, StreamingNeighborhoodAttention2D) for layer in streamed_layers
    )
    assert streamed_layers[0].attention is not streamed_layers[1].attention
    assert streamed_layers[0].seen_indices is not streamed_layers[1].seen_indices

    cache = streaming.get_tile_cache()
    assert cache["net_stats"]["1"]["lost"] == expected_lost

    full_optimizer = torch.optim.SGD(full_module.parameters(), lr=0.025)
    streaming_optimizer = torch.optim.SGD(
        streaming.stream_module.parameters(), lr=0.025
    )

    def run_cycle(cycle):
        full_optimizer.zero_grad(set_to_none=True)
        streaming_optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(8000 + cycle)
        full_input = torch.randn(
            batch, channels, *image_shape, dtype=torch.float32, requires_grad=True
        )
        streaming_input = full_input.detach().clone().requires_grad_(True)
        upstream = torch.randn_like(full_input)

        full_output = full_module(full_input)
        streaming_output = streaming(streaming_input)
        _assert_spatial_regions_match(
            streaming_output,
            full_output,
            query_shape,
            accumulated_radius,
            f"cycle {cycle} StreamingCNN output",
        )

        full_output.backward(upstream)
        streaming.backward(streaming_input, upstream)
        _assert_spatial_regions_match(
            streaming_input.grad,
            full_input.grad,
            query_shape,
            accumulated_radius,
            f"cycle {cycle} StreamingCNN input gradient",
        )
        _assert_named_parameter_state_matches(
            streaming.stream_module, full_module, attribute="grad"
        )

        # backward() must reset both independent de-duplication trackers.  The
        # next cycle then exercises the same StreamingCNN object from scratch.
        for layer in streamed_layers:
            assert layer.input_loc is None
            assert (layer.seen_indices.y, layer.seen_indices.height) == (0, 0)
            assert (layer.seen_indices.x, layer.seen_indices.width) == (0, 0)
            assert layer.seen_indices.sides is None

    run_cycle(0)
    full_optimizer.step()
    streaming_optimizer.step()
    _assert_named_parameter_state_matches(streaming.stream_module, full_module)
    run_cycle(1)


@pytest.mark.parametrize(
    ("active_queries", "region"),
    [
        pytest.param(((6, 3),), "horizontal seam", id="one-query-horizontal-seam"),
        pytest.param(((3, 7),), "vertical seam", id="one-query-vertical-seam"),
        pytest.param(((6, 7),), "seam intersection", id="one-query-seam-intersection"),
        pytest.param(((16, 18),), "final shifted tile", id="one-query-final-shifted-tile"),
        pytest.param(
            ((6, 7), (7, 7)),
            "adjacent tile rows",
            id="queries-in-adjacent-tile-rows",
        ),
        pytest.param(
            ((6, 7), (6, 8)),
            "adjacent tile columns",
            id="queries-in-adjacent-tile-columns",
        ),
    ],
)
def test_shifted_final_tiles_sparse_query_gradients_match_full_frame(
    natten_backend, active_queries, region
):
    """Sparse query gradients must be owned once, including shifted overlaps."""
    torch.manual_seed(2468)
    batch, channels = 1, 8
    image_shape = (17, 19)
    query_shape = (6, 7)
    kernel_size = 3
    radius = (kernel_size - 1) // 2
    tile_shape = (query_shape[0] + 2 * radius, query_shape[1] + 2 * radius)

    # Both axes require a shifted final tile.  Its start precedes the next
    # regular-grid start, so its valid queries overlap those of its predecessor.
    assert image_shape[0] % query_shape[0] != 0
    assert image_shape[1] % query_shape[1] != 0
    final_start = (image_shape[0] - tile_shape[0], image_shape[1] - tile_shape[1])
    previous_start = (query_shape[0], query_shape[1])
    assert final_start[0] < previous_start[0] + query_shape[0]
    assert final_start[1] < previous_start[1] + query_shape[1]

    base = NeighborhoodAttention2D(
        attention=_make_natten(
            natten_backend,
            channels=channels,
            heads=2,
            kernel_size=kernel_size,
            dilation=1,
        )
    )
    full_module = copy.deepcopy(base)
    streaming = StreamingCNN(
        copy.deepcopy(base),
        tile_shape=(batch, channels, *tile_shape),
        copy_to_gpu=True,
    )
    full_input = torch.randn(
        batch, channels, *image_shape, dtype=torch.float32, requires_grad=True
    )
    streaming_input = full_input.detach().clone().requires_grad_(True)
    upstream = torch.zeros_like(full_input)
    channel_weights = torch.linspace(-1.0, 1.0, channels)
    for query_index, (y, x) in enumerate(active_queries, start=1):
        upstream[0, :, y, x] = query_index * channel_weights
    assert torch.count_nonzero(upstream) == len(active_queries) * channels

    full_output = full_module(full_input)
    streaming_output = streaming(streaming_input)
    full_output.backward(upstream)
    streaming.backward(streaming_input, upstream)

    torch.testing.assert_close(
        streaming_input.grad,
        full_input.grad,
        rtol=2e-4,
        atol=2e-5,
        msg=f"input gradient differs for sparse queries at {region}",
    )
    full_parameters = dict(full_module.named_parameters())
    streaming_parameters = dict(streaming.stream_module.named_parameters())
    assert full_parameters.keys() == streaming_parameters.keys()
    for name, full_parameter in full_parameters.items():
        streamed_gradient = streaming_parameters[name].grad
        assert full_parameter.grad is not None, f"full-frame gradient missing for {name!r}"
        assert streamed_gradient is not None, f"streamed gradient missing for {name!r}"
        torch.testing.assert_close(
            streamed_gradient,
            full_parameter.grad,
            rtol=2e-4,
            atol=2e-5,
            msg=f"attention parameter gradient {name!r} differs at {region}",
        )
