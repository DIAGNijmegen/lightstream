import copy
import os

import pytest
import torch
from torch import nn

from lightstream.core.layers import ChannelLayerNorm, StreamingMerge
from lightstream.core.layers.streamingneighborhoodattention import (
    NeighborhoodAttention2D,
    StreamingNeighborhoodAttention2D,
)
from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.core.scnn.utils import Lost
from lightstream.models.nat import (
    ConvDownsampler,
    NCHWNATBlock,
    NCHWNATLayer,
    copy_nhwc_nat_block_to_nchw,
    copy_nhwc_nat_to_nchw,
)

SUPPORTED_NATTEN_VERSION = "0.17.5"
_REL_POS_BIAS_PARAMETER = "rpb"

# NHWC NAT uses Linear kernels while the NCHW implementation uses pointwise
# Conv2d kernels.  Their CUDA reduction order differs enough that comparisons
# crossing that layout/implementation boundary need a modest absolute bound.
# These bounds cover repeated runs of both block depths and both downsampling
# tile geometries below while remaining small relative to substantive errors.
_CROSS_LAYOUT_RTOL = 3e-4
_CROSS_LAYOUT_ATOL = 5e-4

# Full-frame and streamed NCHW execute the same operators.  Keep this comparison
# substantially tighter so layout tolerance cannot hide a streaming defect.
_SAME_LAYOUT_RTOL = 2e-4
_SAME_LAYOUT_ATOL = 2e-5


def test_nchw_conv_downsampler_architecture():
    downsampler = ConvDownsampler(dim=5)

    assert downsampler.reduction.in_channels == 5
    assert downsampler.reduction.out_channels == 10
    assert downsampler.reduction.kernel_size == (3, 3)
    assert downsampler.reduction.stride == (2, 2)
    assert downsampler.reduction.padding == (1, 1)
    assert downsampler.reduction.bias is None
    assert isinstance(downsampler.norm, ChannelLayerNorm)
    assert downsampler.norm.num_channels == 10


@pytest.fixture(scope="session")
def natten_backend():
    """Load only the production NATTEN API for real-backend parity tests."""
    try:
        import natten as backend
    except ImportError:
        if os.environ.get("LIGHTSTREAM_REQUIRE_NATTEN") == "1":
            pytest.fail(
                "the production NAT job requires the real NATTEN backend; "
                "install the pinned 'nat' optional dependency"
            )
        pytest.skip("optional NATTEN backend is not installed")

    if os.environ.get("LIGHTSTREAM_REQUIRE_NATTEN") == "1":
        assert torch.cuda.is_available(), (
            "the production NAT job requires a CUDA-capable runner; "
            "torch.cuda.is_available() is false"
        )
    assert backend.__version__ == SUPPORTED_NATTEN_VERSION, (
        "real-NATTEN parity tests require the v0.17.5-blackwell build; "
        f"found NATTEN {backend.__version__!r}"
    )
    return backend


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
        weight = (
            torch.ones(
                nchw.shape[1],
                1,
                self.kernel_size,
                self.kernel_size,
                device=nchw.device,
                dtype=nchw.dtype,
            )
            / self.kernel_size**2
        )
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
    torch.testing.assert_close(
        actual, expected, rtol=2e-4, atol=2e-5, msg=f"complete {quantity}"
    )
    for region, mask in _region_masks(
        *expected.shape[-2:], query_shape, radius
    ).items():
        assert mask.any(), f"test geometry did not create {region}"
        torch.testing.assert_close(
            actual[..., mask],
            expected[..., mask],
            rtol=2e-4,
            atol=2e-5,
            msg=f"{quantity} differs at {region}",
        )


def _make_natten(natten_backend, channels, heads, kernel_size, dilation):
    # Spell out the production NAT configuration. Keep both attention-weight
    # and projection dropout disabled so tiled and untiled executions are
    # deterministic and directly comparable.
    attention = natten_backend.NeighborhoodAttention2D(
        dim=channels,
        num_heads=heads,
        kernel_size=kernel_size,
        dilation=dilation,
        rel_pos_bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    attention_parameters = dict(attention.named_parameters())
    assert _REL_POS_BIAS_PARAMETER in attention_parameters
    assert attention.attn_drop.p == 0.0
    assert attention.proj_drop.p == 0.0
    return attention.float()


class _LinearMlp(nn.Module):
    """Original NHWC NAT MLP, with dropout deliberately omitted."""

    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, channels)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _NHWCNATLayer(nn.Module):
    """Direct-layout oracle using only original NAT/PyTorch operations."""

    def __init__(self, attention, channels, hidden_channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attn = attention
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = _LinearMlp(channels, hidden_channels)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def _nat_parameter_pairs(reference, nchw):
    """Pair every trainable NAT parameter across the two layouts."""

    for norm_name in ("norm1", "norm2"):
        linear_norm = getattr(reference, norm_name)
        channel_norm = getattr(nchw, norm_name).norm
        yield f"{norm_name}.weight", linear_norm.weight, channel_norm.weight
        yield f"{norm_name}.bias", linear_norm.bias, channel_norm.bias
    nchw_attention = dict(nchw.attn.attention.named_parameters())
    for name, parameter in reference.attn.named_parameters():
        yield f"attn.{name}", parameter, nchw_attention[name]
    for projection_name in ("fc1", "fc2"):
        linear = getattr(reference.mlp, projection_name)
        convolution = getattr(nchw.mlp, projection_name)
        yield f"mlp.{projection_name}.weight", linear.weight, convolution.weight
        yield f"mlp.{projection_name}.bias", linear.bias, convolution.bias


def _nat_block_parameter_pairs(reference, nchw):
    """Pair every attention, normalization, MLP, and downsampling parameter."""

    for index, (reference_layer, nchw_layer) in enumerate(
        zip(reference.blocks, nchw.blocks)
    ):
        for name, reference_parameter, nchw_parameter in _nat_parameter_pairs(
            reference_layer, nchw_layer
        ):
            yield f"blocks.{index}.{name}", reference_parameter, nchw_parameter
    if reference.downsample is not None:
        yield (
            "downsample.reduction.weight",
            reference.downsample.reduction.weight,
            nchw.downsample.reduction.weight,
        )
        yield (
            "downsample.norm.weight",
            reference.downsample.norm.weight,
            nchw.downsample.norm.norm.weight,
        )
        yield (
            "downsample.norm.bias",
            reference.downsample.norm.bias,
            nchw.downsample.norm.norm.bias,
        )


def _linear_shaped(value, reference):
    """Remove the spatial singleton axes of a pointwise-convolution tensor."""

    if value.ndim == reference.ndim + 2:
        return value[:, :, 0, 0]
    return value


def test_nchw_wrapper_and_manual_halo_oracle_match_values_gradients_and_step():
    torch.manual_seed(5)
    reference = NeighborhoodAttention2D(
        attention=_LocalNHWCBackend(kernel_size=5, dilation=2)
    )
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
    direct = _make_natten(
        natten_backend, channels=8, heads=2, kernel_size=3, dilation=1
    )
    wrapped = NeighborhoodAttention2D(attention=copy.deepcopy(direct))
    direct_input = torch.randn(1, 11, 13, 8, dtype=torch.float32, requires_grad=True)
    wrapped_input = (
        direct_input.detach().permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    )
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
    _assert_spatial_regions_match(
        manual_output, full_output, query_shape, radius, "manual output"
    )
    _assert_spatial_regions_match(
        streaming_output, full_output, query_shape, radius, "StreamingCNN output"
    )

    full_output.backward(upstream)
    manual_output.backward(upstream)
    streaming.backward(streaming_input, upstream)
    for execution, module in (
        ("full-frame", full_module),
        ("manual-tiled", manual_module),
        ("StreamingCNN", streaming.stream_module),
    ):
        attention_parameters = dict(module.attention.named_parameters())
        assert _REL_POS_BIAS_PARAMETER in attention_parameters
        assert (
            attention_parameters[_REL_POS_BIAS_PARAMETER].grad is not None
        ), f"{execution} relative-position-bias gradient is missing"
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
        pytest.param(((0, 9),), "true edge", id="one-query-true-edge"),
        pytest.param(((0, 0),), "corner", id="one-query-corner"),
        pytest.param(((6, 3),), "horizontal seam", id="one-query-horizontal-seam"),
        pytest.param(((3, 7),), "vertical seam", id="one-query-vertical-seam"),
        pytest.param(((6, 7),), "seam intersection", id="one-query-seam-intersection"),
        pytest.param(
            ((16, 18),), "final shifted tile", id="one-query-final-shifted-tile"
        ),
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
    relative_bias_name = f"attention.{_REL_POS_BIAS_PARAMETER}"
    assert relative_bias_name in full_parameters
    assert full_parameters[relative_bias_name].grad is not None
    assert streaming_parameters[relative_bias_name].grad is not None
    for name, full_parameter in full_parameters.items():
        streamed_gradient = streaming_parameters[name].grad
        assert (
            full_parameter.grad is not None
        ), f"full-frame gradient missing for {name!r}"
        assert streamed_gradient is not None, f"streamed gradient missing for {name!r}"
        torch.testing.assert_close(
            streamed_gradient,
            full_parameter.grad,
            rtol=2e-4,
            atol=2e-5,
            msg=f"attention parameter gradient {name!r} differs at {region}",
        )


def test_channel_layer_norm_affine_gradients_keep_attention_halo_contributions():
    """LayerNorm affine gradients include halo dependencies of owned queries."""
    torch.manual_seed(8642)
    batch, channels = 1, 4
    image_shape = (17, 19)
    query_shape = (6, 7)
    radius = 1
    tile_shape = tuple(query + 2 * radius for query in query_shape)

    assert all(image % query for image, query in zip(image_shape, query_shape))
    final_start = tuple(image - tile for image, tile in zip(image_shape, tile_shape))
    previous_start = query_shape
    assert all(
        final < previous + query
        for final, previous, query in zip(final_start, previous_start, query_shape)
    )
    base = nn.Sequential(
        ChannelLayerNorm(channels),
        NeighborhoodAttention2D(
            attention=_LocalNHWCBackend(channels=channels, kernel_size=3)
        ),
    )
    full_module = copy.deepcopy(base)
    streaming = StreamingCNN(
        copy.deepcopy(base),
        tile_shape=(batch, channels, *tile_shape),
        copy_to_gpu=False,
    )
    full_input = torch.randn(batch, channels, *image_shape, requires_grad=True)
    streaming_input = full_input.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(full_input)

    full_module(full_input).backward(upstream)
    streaming(streaming_input)
    streaming.backward(streaming_input, upstream)

    streaming_norm = streaming.stream_module[0]
    for name in ("weight", "bias"):
        torch.testing.assert_close(
            getattr(streaming_norm.norm, name).grad,
            getattr(full_module[0].norm, name).grad,
            rtol=2e-4,
            atol=2e-5,
            msg=f"LayerNorm {name} gradient lost an attention halo contribution",
        )


def test_complete_nchw_nat_layer_matches_nhwc_reference_and_streaming(natten_backend):
    """A complete kernel-7 NAT layer matches over two streaming cycles."""

    torch.manual_seed(97531)
    batch, channels, hidden_channels, heads = 1, 8, 24, 2
    kernel_size = 7
    radius = (kernel_size - 1) // 2
    image_shape = (16, 19)
    query_shape = (5, 7)
    tile_shape = tuple(query + 2 * radius for query in query_shape)

    # Neither axis ends on the regular query grid. StreamingCNN must shift the
    # final physical tile back, creating overlap that backward de-duplicates.
    assert all(image % query for image, query in zip(image_shape, query_shape))
    assert all(image > tile for image, tile in zip(image_shape, tile_shape))

    reference = _NHWCNATLayer(
        _make_natten(natten_backend, channels, heads, kernel_size, dilation=1),
        channels,
        hidden_channels,
    ).float()
    nchw_source = NCHWNATLayer(
        _make_natten(natten_backend, channels, heads, kernel_size, dilation=1),
        channels,
        hidden_channels,
    ).float()
    copy_nhwc_nat_to_nchw(reference, nchw_source)
    full_nchw = copy.deepcopy(nchw_source)
    streaming = StreamingCNN(
        nchw_source,
        tile_shape=(batch, channels, *tile_shape),
        copy_to_gpu=True,
    )

    # NATTEN owns dropout modules, but the fixture disables them so tiled and
    # untiled executions remain deterministic. The MLPs omit dropout entirely.
    for model in (reference, full_nchw, streaming.stream_module):
        dropouts = [
            module for module in model.modules() if isinstance(module, nn.Dropout)
        ]
        assert all(module.p == 0.0 for module in dropouts)
    assert not any(isinstance(module, nn.Dropout) for module in reference.mlp.modules())
    assert not any(isinstance(module, nn.Dropout) for module in full_nchw.mlp.modules())

    # Both residual additions remain explicit module boundaries after conversion.
    assert isinstance(nchw_source.merge1, StreamingMerge)
    assert isinstance(nchw_source.merge2, StreamingMerge)
    assert nchw_source.merge1.mode == "add"
    assert nchw_source.merge2.mode == "add"

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.015)
    full_optimizer = torch.optim.SGD(full_nchw.parameters(), lr=0.015)
    streaming_optimizer = torch.optim.SGD(
        streaming.stream_module.parameters(), lr=0.015
    )

    for cycle in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        full_optimizer.zero_grad(set_to_none=True)
        streaming_optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(11100 + cycle)
        reference_input = torch.randn(
            batch, *image_shape, channels, dtype=torch.float32, requires_grad=True
        )
        nchw_input = (
            reference_input.detach()
            .permute(0, 3, 1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        streaming_input = nchw_input.detach().clone().requires_grad_(True)
        upstream = torch.randn_like(reference_input)

        reference_output = reference(reference_input)
        full_output = full_nchw(nchw_input)
        streaming_output = streaming(streaming_input)
        torch.testing.assert_close(
            full_output,
            reference_output.permute(0, 3, 1, 2),
            rtol=_CROSS_LAYOUT_RTOL,
            atol=_CROSS_LAYOUT_ATOL,
            msg=f"cycle {cycle} full-frame NAT output",
        )
        torch.testing.assert_close(
            streaming_output,
            full_output,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            msg=f"cycle {cycle} complete NAT output",
        )

        reference_output.backward(upstream)
        nchw_upstream = upstream.permute(0, 3, 1, 2).contiguous()
        full_output.backward(nchw_upstream)
        streaming.backward(streaming_input, nchw_upstream)
        torch.testing.assert_close(
            nchw_input.grad,
            reference_input.grad.permute(0, 3, 1, 2),
            rtol=_CROSS_LAYOUT_RTOL,
            atol=_CROSS_LAYOUT_ATOL,
            msg=f"cycle {cycle} full-frame NAT input gradient",
        )
        torch.testing.assert_close(
            streaming_input.grad,
            nchw_input.grad,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            msg=f"cycle {cycle} streaming NAT input gradient",
        )
        full_pairs = list(_nat_parameter_pairs(reference, full_nchw))
        streaming_pairs = list(_nat_parameter_pairs(reference, streaming.stream_module))
        for name, reference_parameter, full_parameter in full_pairs:
            assert reference_parameter.grad is not None
            assert full_parameter.grad is not None
            torch.testing.assert_close(
                _linear_shaped(full_parameter.grad, reference_parameter.grad),
                reference_parameter.grad,
                rtol=_CROSS_LAYOUT_RTOL,
                atol=_CROSS_LAYOUT_ATOL,
                msg=f"cycle {cycle} full-frame parameter gradient {name}",
            )
        for (name, _, full_parameter), (streaming_name, _, streaming_parameter) in zip(
            full_pairs, streaming_pairs
        ):
            assert name == streaming_name
            assert (
                streaming_parameter.grad is not None
            ), f"missing NCHW gradient for {name}"
            torch.testing.assert_close(
                streaming_parameter.grad,
                full_parameter.grad,
                rtol=_SAME_LAYOUT_RTOL,
                atol=_SAME_LAYOUT_ATOL,
                msg=f"cycle {cycle} streaming parameter gradient {name}",
            )
        # Exactly one optimizer step is compared. The second cycle verifies
        # reuse of the same StreamingCNN and its reset state after that step.
        if cycle == 0:
            reference_optimizer.step()
            full_optimizer.step()
            streaming_optimizer.step()
            full_pairs = list(_nat_parameter_pairs(reference, full_nchw))
            streaming_pairs = list(
                _nat_parameter_pairs(reference, streaming.stream_module)
            )
            for name, reference_parameter, full_parameter in full_pairs:
                torch.testing.assert_close(
                    _linear_shaped(full_parameter, reference_parameter),
                    reference_parameter,
                    rtol=_CROSS_LAYOUT_RTOL,
                    atol=_CROSS_LAYOUT_ATOL,
                    msg=f"full-frame optimizer-updated parameter {name}",
                )
            for (name, _, full_parameter), (
                streaming_name,
                _,
                streaming_parameter,
            ) in zip(full_pairs, streaming_pairs):
                assert name == streaming_name
                torch.testing.assert_close(
                    streaming_parameter,
                    full_parameter,
                    rtol=_SAME_LAYOUT_RTOL,
                    atol=_SAME_LAYOUT_ATOL,
                    msg=f"streaming optimizer-updated parameter {name}",
                )


@pytest.mark.parametrize(
    "dilations",
    [
        pytest.param((1, 2), id="depth-2"),
        pytest.param((1, 2, 1), id="nat-mini-first-stage-depth-3"),
    ],
)
def test_complete_nat_block_matches_reference_streaming_and_reset(
    natten_backend, dilations
):
    """Production NAT blocks preserve values, gradients, updates, and stream state."""

    # Importing the NHWC model performs the package-version check, so do this
    # only after the optional real-backend fixture has selected the test.
    from lightstream.models.nat.nat import NATBlock

    torch.manual_seed(24680 + len(dilations))
    batch, channels, heads = 1, 8, 2
    kernel_size, mlp_ratio = 7, 3
    radii = tuple(dilation * (kernel_size - 1) // 2 for dilation in dilations)
    accumulated_radius = sum(radii)
    query_shape = (5, 7)
    tile_hw = tuple(query + 2 * accumulated_radius for query in query_shape)
    image_shape = tuple(
        tile + 3 * query + 1 for tile, query in zip(tile_hw, query_shape)
    )

    reference = NATBlock(
        dim=channels,
        depth=len(dilations),
        num_heads=heads,
        kernel_size=kernel_size,
        dilations=dilations,
        downsample=False,
        mlp_ratio=mlp_ratio,
        drop_path=0.0,
    ).float()
    nchw_source = NCHWNATBlock(
        channels=channels,
        depth=len(dilations),
        num_heads=heads,
        kernel_size=kernel_size,
        dilations=dilations,
        downsample=False,
        mlp_ratio=mlp_ratio,
    ).float()
    copy_nhwc_nat_block_to_nchw(reference, nchw_source)
    full_nchw = copy.deepcopy(nchw_source)
    streaming = StreamingCNN(
        nchw_source, tile_shape=(batch, channels, *tile_hw), copy_to_gpu=True
    )

    streamed_attentions = [layer.attn for layer in streaming.stream_module.blocks]
    assert all(
        isinstance(item, StreamingNeighborhoodAttention2D)
        for item in streamed_attentions
    )
    assert len({id(item.seen_indices) for item in streamed_attentions}) == len(
        dilations
    )
    assert reference.downsample is None and full_nchw.downsample is None

    # Confirm all parameter families requested by the checkpoint conversion are present.
    for layer in reference.blocks:
        names = dict(layer.named_parameters())
        assert {
            "norm1.weight",
            "norm1.bias",
            "attn.qkv.weight",
            "attn.proj.weight",
            "mlp.fc1.weight",
            "mlp.fc2.weight",
        } <= names.keys()
        assert f"attn.{_REL_POS_BIAS_PARAMETER}" in names

    cache = streaming.get_tile_cache()
    cumulative = 0
    for index, radius in enumerate(radii):
        cumulative += radius
        stats = cache["net_stats"][f"blocks.{index}.attn"]
        assert stats["directional_spatial_support"] == Lost(
            radius, radius, radius, radius
        )
        assert stats["lost"] == Lost(cumulative, cumulative, cumulative, cumulative)

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.015)
    full_optimizer = torch.optim.SGD(full_nchw.parameters(), lr=0.015)
    streaming_optimizer = torch.optim.SGD(
        streaming.stream_module.parameters(), lr=0.015
    )

    for cycle in range(2):
        for optimizer in (reference_optimizer, full_optimizer, streaming_optimizer):
            optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(26000 + cycle)
        reference_input = torch.randn(
            batch, *image_shape, channels, dtype=torch.float32, requires_grad=True
        )
        full_input = (
            reference_input.detach()
            .permute(0, 3, 1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        streaming_input = full_input.detach().clone().requires_grad_(True)
        upstream = torch.randn_like(reference_input)

        reference_output = reference(reference_input)
        full_output = full_nchw(full_input)
        streaming_output = streaming(streaming_input)
        expected_output = reference_output.permute(0, 3, 1, 2)
        torch.testing.assert_close(
            full_output,
            expected_output,
            rtol=_CROSS_LAYOUT_RTOL,
            atol=_CROSS_LAYOUT_ATOL,
        )
        torch.testing.assert_close(
            streaming_output,
            full_output,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
        )

        reference_output.backward(upstream)
        nchw_upstream = upstream.permute(0, 3, 1, 2).contiguous()
        full_output.backward(nchw_upstream)
        streaming.backward(streaming_input, nchw_upstream)
        expected_input_grad = reference_input.grad.permute(0, 3, 1, 2)
        torch.testing.assert_close(
            full_input.grad,
            expected_input_grad,
            rtol=_CROSS_LAYOUT_RTOL,
            atol=_CROSS_LAYOUT_ATOL,
        )
        torch.testing.assert_close(
            streaming_input.grad,
            full_input.grad,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
        )

        for layer_index, reference_layer in enumerate(reference.blocks):
            full_pairs = list(
                _nat_parameter_pairs(reference_layer, full_nchw.blocks[layer_index])
            )
            streaming_pairs = list(
                _nat_parameter_pairs(
                    reference_layer, streaming.stream_module.blocks[layer_index]
                )
            )
            assert {name for name, _, _ in full_pairs} == dict(
                reference_layer.named_parameters()
            ).keys()
            for name, reference_parameter, full_parameter in full_pairs:
                assert reference_parameter.grad is not None
                assert full_parameter.grad is not None
                torch.testing.assert_close(
                    _linear_shaped(full_parameter.grad, reference_parameter.grad),
                    reference_parameter.grad,
                    rtol=_CROSS_LAYOUT_RTOL,
                    atol=_CROSS_LAYOUT_ATOL,
                    msg=f"cycle {cycle} full-frame parameter gradient {name}",
                )
            for full_pair, streaming_pair in zip(full_pairs, streaming_pairs):
                name, _, full_parameter = full_pair
                streaming_name, _, streaming_parameter = streaming_pair
                assert name == streaming_name
                assert streaming_parameter.grad is not None
                torch.testing.assert_close(
                    streaming_parameter.grad,
                    full_parameter.grad,
                    rtol=_SAME_LAYOUT_RTOL,
                    atol=_SAME_LAYOUT_ATOL,
                    msg=f"cycle {cycle} streaming parameter gradient {name}",
                )

        for attention in streamed_attentions:
            assert attention.input_loc is None
            assert (attention.seen_indices.y, attention.seen_indices.height) == (0, 0)
            assert (attention.seen_indices.x, attention.seen_indices.width) == (0, 0)
            assert attention.seen_indices.sides is None

        # Step once, then reuse this exact StreamingCNN for a second cycle.
        if cycle == 0:
            for optimizer in (reference_optimizer, full_optimizer, streaming_optimizer):
                optimizer.step()
            for layer_index, reference_layer in enumerate(reference.blocks):
                full_pairs = list(
                    _nat_parameter_pairs(reference_layer, full_nchw.blocks[layer_index])
                )
                streaming_pairs = list(
                    _nat_parameter_pairs(
                        reference_layer, streaming.stream_module.blocks[layer_index]
                    )
                )
                for name, reference_parameter, full_parameter in full_pairs:
                    torch.testing.assert_close(
                        _linear_shaped(full_parameter, reference_parameter),
                        reference_parameter,
                        rtol=_CROSS_LAYOUT_RTOL,
                        atol=_CROSS_LAYOUT_ATOL,
                        msg=f"full-frame optimizer-updated parameter {name}",
                    )
                for full_pair, streaming_pair in zip(full_pairs, streaming_pairs):
                    name, _, full_parameter = full_pair
                    streaming_name, _, streaming_parameter = streaming_pair
                    assert name == streaming_name
                    torch.testing.assert_close(
                        streaming_parameter,
                        full_parameter,
                        rtol=_SAME_LAYOUT_RTOL,
                        atol=_SAME_LAYOUT_ATOL,
                        msg=f"streaming optimizer-updated parameter {name}",
                    )


@pytest.mark.parametrize(
    ("tile_hw", "image_hw", "expected_downsample_lost"),
    [
        pytest.param(
            (18, 20),
            (27, 31),
            Lost(top=4, left=4, bottom=3, right=3),
            id="stride-aligned-tile",
        ),
        pytest.param(
            (19, 21),
            (29, 33),
            Lost(top=4, left=4, bottom=4, right=4),
            id="stride-unaligned-tile",
        ),
    ],
)
def test_nat_block_downsampler_matches_reference_and_streaming(
    natten_backend, tile_hw, image_hw, expected_downsample_lost
):
    """An odd, non-square NAT stage remains exact across its stride-2 boundary."""

    from lightstream.models.nat.nat import NATBlock

    torch.manual_seed(31001)
    batch, channels, heads = 1, 8, 2
    reference = NATBlock(
        dim=channels,
        depth=2,
        num_heads=heads,
        kernel_size=7,
        dilations=(1, 1),
        downsample=True,
        mlp_ratio=3,
        drop_path=0.0,
    ).float()
    nchw_source = NCHWNATBlock(
        channels=channels,
        depth=2,
        num_heads=heads,
        kernel_size=7,
        dilations=(1, 1),
        downsample=True,
        mlp_ratio=3,
    ).float()
    copy_nhwc_nat_block_to_nchw(reference, nchw_source)
    full_nchw = copy.deepcopy(nchw_source)
    streaming = StreamingCNN(
        nchw_source,
        tile_shape=(batch, channels, *tile_hw),
        copy_to_gpu=True,
    )

    assert image_hw[0] % 2 == 1 and image_hw[1] % 2 == 1
    assert image_hw[0] != image_hw[1]
    reduction_stats = streaming.get_tile_cache()["net_stats"]["downsample.reduction"]
    norm_stats = streaming.get_tile_cache()["net_stats"]["downsample.norm"]
    assert tuple(reduction_stats["stride"]) == (0, 2, 2)
    assert reduction_stats["output_stride"].tolist() == [1, 1, 1]
    assert reduction_stats["lost"] == expected_downsample_lost
    assert norm_stats["stride"].tolist() == [1, 1, 1]
    assert norm_stats["output_stride"].tolist() == [1, 2, 2]
    assert norm_stats["lost"] == expected_downsample_lost
    assert streaming.output_stride.tolist() == [1, 2, 2]

    implementations = (reference, full_nchw, streaming.stream_module)
    optimizers = tuple(
        torch.optim.SGD(module.parameters(), lr=0.015) for module in implementations
    )
    reference_input = torch.randn(
        batch, *image_hw, channels, dtype=torch.float32, requires_grad=True
    )
    full_input = (
        reference_input.detach().permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    )
    streaming_input = full_input.detach().clone().requires_grad_(True)

    reference_output = reference(reference_input)
    full_output = full_nchw(full_input)
    streaming_output = streaming(streaming_input)
    expected_output = reference_output.permute(0, 3, 1, 2)
    assert expected_output.shape == (
        batch,
        2 * channels,
        (image_hw[0] + 1) // 2,
        (image_hw[1] + 1) // 2,
    )
    torch.testing.assert_close(
        full_output,
        expected_output,
        rtol=_CROSS_LAYOUT_RTOL,
        atol=_CROSS_LAYOUT_ATOL,
    )
    torch.testing.assert_close(
        streaming_output,
        full_output,
        rtol=_SAME_LAYOUT_RTOL,
        atol=_SAME_LAYOUT_ATOL,
    )

    upstream = torch.randn_like(reference_output)
    reference_output.backward(upstream)
    nchw_upstream = upstream.permute(0, 3, 1, 2).contiguous()
    full_output.backward(nchw_upstream)
    streaming.backward(streaming_input, nchw_upstream)
    expected_input_grad = reference_input.grad.permute(0, 3, 1, 2)
    torch.testing.assert_close(
        full_input.grad,
        expected_input_grad,
        rtol=_CROSS_LAYOUT_RTOL,
        atol=_CROSS_LAYOUT_ATOL,
    )
    torch.testing.assert_close(
        streaming_input.grad,
        full_input.grad,
        rtol=_SAME_LAYOUT_RTOL,
        atol=_SAME_LAYOUT_ATOL,
    )

    reference_names = set(dict(reference.named_parameters()))
    full_pairs = list(_nat_block_parameter_pairs(reference, full_nchw))
    streaming_pairs = list(
        _nat_block_parameter_pairs(reference, streaming.stream_module)
    )
    assert {name for name, _, _ in full_pairs} == reference_names
    for name, reference_parameter, full_parameter in full_pairs:
        assert reference_parameter.grad is not None
        assert full_parameter.grad is not None
        torch.testing.assert_close(
            _linear_shaped(full_parameter.grad, reference_parameter.grad),
            reference_parameter.grad,
            rtol=_CROSS_LAYOUT_RTOL,
            atol=_CROSS_LAYOUT_ATOL,
            msg=f"full-frame parameter gradient {name}",
        )
    for full_pair, streaming_pair in zip(full_pairs, streaming_pairs):
        name, _, full_parameter = full_pair
        streaming_name, _, streaming_parameter = streaming_pair
        assert name == streaming_name
        assert streaming_parameter.grad is not None
        torch.testing.assert_close(
            streaming_parameter.grad,
            full_parameter.grad,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            msg=f"streaming parameter gradient {name}",
        )

    for optimizer in optimizers:
        optimizer.step()
    full_pairs = list(_nat_block_parameter_pairs(reference, full_nchw))
    streaming_pairs = list(
        _nat_block_parameter_pairs(reference, streaming.stream_module)
    )
    for name, reference_parameter, full_parameter in full_pairs:
        torch.testing.assert_close(
            _linear_shaped(full_parameter, reference_parameter),
            reference_parameter,
            rtol=_CROSS_LAYOUT_RTOL,
            atol=_CROSS_LAYOUT_ATOL,
            msg=f"full-frame optimizer-updated parameter {name}",
        )
    for full_pair, streaming_pair in zip(full_pairs, streaming_pairs):
        name, _, full_parameter = full_pair
        streaming_name, _, streaming_parameter = streaming_pair
        assert name == streaming_name
        torch.testing.assert_close(
            streaming_parameter,
            full_parameter,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            msg=f"streaming optimizer-updated parameter {name}",
        )
