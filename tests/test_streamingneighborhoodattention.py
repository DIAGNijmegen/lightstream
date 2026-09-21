import copy
import os

import pytest
import torch
from torch import nn

from lightstream.core.layers import (
    ChannelLayerNorm,
    StreamingLayerScale,
    StreamingMerge,
)
from lightstream.core.layers.streamingneighborhoodattention import (
    NeighborhoodAttention2D,
    StreamingNeighborhoodAttention2D,
)
from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.core.scnn.utils import Lost
from lightstream.models.nat import (
    ConvDownsampler,
    NCHWConvTokenizer,
    NCHWNAT,
    NCHWNATBlock,
    NCHWNATLayer,
    convert_nchw_nat_state_dict,
    convert_nhwc_nat_state_dict,
    copy_nhwc_nat_block_to_nchw,
    copy_nhwc_nat_to_nchw,
    copy_nhwc_conv_tokenizer_to_nchw,
)

SUPPORTED_NATTEN_VERSION = "0.17.5"
_REL_POS_BIAS_PARAMETER = "rpb"

# NHWC NAT uses Linear kernels while the NCHW implementation uses pointwise
# Conv2d kernels. Their CUDA reduction order affects the three quantities below
# differently, so do not collapse these back into one cross-layout tolerance.
# The bounds include a small margin over repeated runs on the pinned NATTEN
# 0.17.5/CUDA job, for depth 2 and 3 blocks and both downsampler tile shapes.
_CROSS_LAYOUT_OUTPUT_RTOL = 3e-4
_CROSS_LAYOUT_OUTPUT_ATOL = 5e-4
_CROSS_LAYOUT_INPUT_GRAD_RTOL = 4e-4
_CROSS_LAYOUT_INPUT_GRAD_ATOL = 7e-4
# Parameter gradients accumulate reductions over every spatial query. The
# pinned-job diagnostics peaked just below 1.7e-3, hence the 2e-3 absolute
# bound. Relative error away from zero remained below 4e-4.
_CROSS_LAYOUT_PARAMETER_GRAD_RTOL = 5e-4
_CROSS_LAYOUT_PARAMETER_GRAD_ATOL = 2e-3

# The four-stage CUDA integration case is substantially deeper than the unit
# cases covered by the general cross-layout bounds above.  Measurements on the
# pinned NATTEN 0.17.5 CUDA runner, including both forward/backward cycles,
# peaked at 8.5155e-4 / 2.76e-4 (absolute / relative) for features,
# 5.43e-4 / 3.61e-4 for image gradients, and 1.68e-3 / 4.18e-4 across all named
# parameter gradients.  The single optimizer step produced parameter
# differences of at most 1.69e-5 / 4.17e-4.  Keep a small, explicit margin over
# those complete-model maxima without relaxing the tolerances of smaller tests.
_COMPLETE_MODEL_FEATURE_RTOL = 3e-4
_COMPLETE_MODEL_FEATURE_ATOL = 1e-3
_COMPLETE_MODEL_IMAGE_GRAD_RTOL = 4e-4
_COMPLETE_MODEL_IMAGE_GRAD_ATOL = 7e-4
_COMPLETE_MODEL_PARAMETER_GRAD_RTOL = 5e-4
_COMPLETE_MODEL_PARAMETER_GRAD_ATOL = 2e-3
_COMPLETE_MODEL_PARAMETER_RTOL = 5e-4
_COMPLETE_MODEL_PARAMETER_ATOL = 2e-5

# Full-frame and streamed NCHW execute the same operators.  Keep this comparison
# substantially tighter so layout tolerance cannot hide a streaming defect.
_SAME_LAYOUT_RTOL = 2e-4
_SAME_LAYOUT_ATOL = 2e-5

# The heavy four-stage CUDA case covers shifted tiles at the complete model's
# full receptive field and therefore needs its own streamed-feature bound.  On
# the pinned NATTEN 0.17.5/CUDA runner, repeated initial and post-optimizer
# cycles both remained at or below 6.7281723e-4 maximum absolute error (the
# reported 0.4239 relative maximum was at numerical zero).  The 1e-3 absolute
# bound leaves a limited ~49% margin over that repeatable maximum, while the
# relative bound remains as strict as the focused same-layout streaming tests.
_COMPLETE_MODEL_STREAMED_FEATURE_RTOL = 2e-4
_COMPLETE_MODEL_STREAMED_FEATURE_ATOL = 1e-3

# Backward replay has its own CUDA reduction order; these bounds are therefore
# deliberately independent of the feature-map bound above.  Across the initial
# and post-optimizer cycles, and both the original and cache-restored streamers,
# the measured streamed/full NCHW maxima were 3.0517578e-5 / 1.7346655e-4
# (absolute / relative) for image gradients and 4.8828125e-4 / 1.9371508e-4
# across all named parameter gradients.  Only the absolute bounds need to be
# wider than the focused same-layout bounds.
_COMPLETE_MODEL_STREAMED_IMAGE_GRAD_RTOL = _SAME_LAYOUT_RTOL
_COMPLETE_MODEL_STREAMED_IMAGE_GRAD_ATOL = 5e-5
_COMPLETE_MODEL_STREAMED_PARAMETER_GRAD_RTOL = _SAME_LAYOUT_RTOL
_COMPLETE_MODEL_STREAMED_PARAMETER_GRAD_ATOL = 7e-4


def _assert_close_with_diagnostics(
    actual,
    expected,
    *,
    rtol,
    atol,
    quantity,
    parameter_name=None,
    cycle=None,
    streaming=None,
):
    """Assert closeness and report useful CUDA parity diagnostics on failure."""
    actual_detached = actual.detach()
    expected_detached = expected.detach()
    absolute_difference = (actual_detached - expected_detached).abs()
    maximum_absolute_difference = absolute_difference.max().item()
    maximum_magnitude = torch.maximum(
        actual_detached.abs().max(), expected_detached.abs().max()
    ).item()

    # Relative errors at numerical zero are uninformative and can be enormous.
    # Use the assertion's absolute bound as the definition of "near zero".
    denominator = torch.maximum(actual_detached.abs(), expected_detached.abs())
    away_from_zero = denominator > max(atol, torch.finfo(denominator.dtype).eps)
    if away_from_zero.any():
        maximum_relative_difference = (
            (absolute_difference[away_from_zero] / denominator[away_from_zero])
            .max()
            .item()
        )
    else:
        maximum_relative_difference = 0.0

    context = [quantity]
    if parameter_name is not None:
        context.append(f"parameter={parameter_name!r}")
    if cycle is not None:
        context.append(f"cycle={cycle}")
    if actual_detached.ndim >= 4:
        flat_index = int(absolute_difference.argmax().item())
        maximum_index = []
        for size in reversed(absolute_difference.shape):
            maximum_index.append(flat_index % size)
            flat_index //= size
        maximum_index.reverse()
        context.append(
            "max_diff_index="
            f"(channel={maximum_index[-3]}, y={maximum_index[-2]}, "
            f"x={maximum_index[-1]})"
        )
    if streaming is not None and actual_detached.ndim >= 4:
        starts = [(int(y), int(x)) for y, x, _ in streaming._last_forward_tiles]
        context.append(f"input_tile_starts={starts}")

        valid_heights, valid_widths = streaming._compute_valid_output_sizes()
        step_y, step_x = streaming._compute_valid_input_step(
            valid_heights, valid_widths
        )
        stride = streaming._output_stride_per_output[0]
        feature_y, feature_x = maximum_index[-2:]
        input_y = feature_y * int(stride[1])
        input_x = feature_x * int(stride[2])
        rows = sorted({y for y, _ in starts})
        columns = sorted({x for _, x in starts})

        def boundary_kind(coordinate, axis_starts, step, valid_size, axis_stride):
            boundaries = set()
            shifted_boundaries = set()
            for position, start in enumerate(axis_starts):
                output_start = start // axis_stride
                output_end = output_start + valid_size
                boundaries.update((output_start, output_end))
                if start != position * step:
                    shifted_boundaries.update((output_start, output_end))
            if coordinate not in boundaries:
                return "not-on-tile-boundary"
            return "shifted" if coordinate in shifted_boundaries else "regular"

        context.append(f"max_diff_input_coordinate=(y={input_y}, x={input_x})")
        context.append(
            "tile_boundary="
            f"(y={boundary_kind(feature_y, rows, step_y, valid_heights[0], int(stride[1]))}, "
            f"x={boundary_kind(feature_x, columns, step_x, valid_widths[0], int(stride[2]))})"
        )
    message = (
        f"{', '.join(context)}; max_abs_diff={maximum_absolute_difference:.9g}; "
        f"max_rel_diff_away_from_zero={maximum_relative_difference:.9g}; "
        f"max_tensor_magnitude={maximum_magnitude:.9g}"
    )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=message)


def _assert_identical_state_dict(full_nchw, streaming):
    """Require exact same-layout parameter and persistent-buffer state."""
    full_state = full_nchw.state_dict()
    streaming_state = streaming.stream_module.state_dict()
    assert full_state.keys() == streaming_state.keys()
    for name, full_value in full_state.items():
        torch.testing.assert_close(
            streaming_state[name],
            full_value,
            rtol=0,
            atol=0,
            msg=f"same-layout state value {name!r} differs",
        )


def test_close_diagnostics_identify_parameter_cycle_and_error_scales():
    with pytest.raises(AssertionError) as error:
        _assert_close_with_diagnostics(
            torch.tensor([0.0, 2.0]),
            torch.tensor([0.0, 1.0]),
            rtol=0,
            atol=1e-6,
            quantity="accumulated parameter gradient",
            parameter_name="blocks.1.attn.qkv.weight",
            cycle=2,
        )

    message = str(error.value)
    assert "parameter='blocks.1.attn.qkv.weight'" in message
    assert "cycle=2" in message
    assert "max_abs_diff=1" in message
    assert "max_rel_diff_away_from_zero=0.5" in message
    assert "max_tensor_magnitude=2" in message


def test_complete_nat_state_conversion_maps_wrappers_and_mlp_kernels():
    original = {
        "patch_embed.norm.weight": torch.randn(8),
        "levels.0.blocks.0.norm1.bias": torch.randn(8),
        "levels.0.blocks.0.attn.qkv.weight": torch.randn(24, 8),
        "levels.0.blocks.0.gamma1": torch.randn(8),
        "levels.0.blocks.0.gamma2": torch.randn(8),
        "levels.0.blocks.0.mlp.fc1.weight": torch.randn(16, 8),
        "levels.0.downsample.norm.weight": torch.randn(16),
        "norm.bias": torch.randn(64),
    }

    converted = convert_nhwc_nat_state_dict(original)

    assert set(converted) == {
        "patch_embed.norm.norm.weight",
        "levels.0.blocks.0.norm1.norm.bias",
        "levels.0.blocks.0.attn.attention.qkv.weight",
        "levels.0.blocks.0.gamma1.weight",
        "levels.0.blocks.0.gamma2.weight",
        "levels.0.blocks.0.mlp.fc1.weight",
        "levels.0.downsample.norm.norm.weight",
        "norm.norm.bias",
    }
    assert converted["levels.0.blocks.0.mlp.fc1.weight"].shape == (16, 8, 1, 1)
    assert converted["levels.0.blocks.0.gamma1.weight"].shape == (1, 8, 1, 1)
    assert converted["levels.0.blocks.0.gamma2.weight"].shape == (1, 8, 1, 1)
    restored = convert_nchw_nat_state_dict(converted)
    assert restored.keys() == original.keys()
    for name in original:
        torch.testing.assert_close(restored[name], original[name])


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


@pytest.mark.parametrize("tile_hw", [(20, 24), (19, 23)])
def test_conv_tokenizer_matches_nhwc_full_frame_and_streaming(natten_backend, tile_hw):
    """The two tokenizer strides retain forward, backward, and update parity."""

    from lightstream.models.nat.nat import ConvTokenizer

    torch.manual_seed(1701)
    batch, in_channels, embed_dim = 1, 3, 8
    image_hw = (31, 37)
    reference = ConvTokenizer(
        in_chans=in_channels, embed_dim=embed_dim, norm_layer=nn.LayerNorm
    ).double()
    reference.proj[0].weight.requires_grad_(False)

    source = copy_nhwc_conv_tokenizer_to_nchw(
        reference, NCHWConvTokenizer(in_channels, embed_dim)
    )
    full_nchw = copy.deepcopy(source)
    streaming = StreamingCNN(
        source,
        tile_shape=(batch, in_channels, *tile_hw),
        copy_to_gpu=True,
    )

    stats = streaming.get_tile_cache()["net_stats"]
    first_stats, second_stats, norm_stats = (
        stats["proj.0"],
        stats["proj.1"],
        stats["norm"],
    )
    assert tuple(first_stats["stride"]) == (0, 2, 2)
    assert first_stats["output_stride"].tolist() == [1, 1, 1]
    assert tuple(second_stats["stride"]) == (0, 2, 2)
    assert second_stats["output_stride"].tolist() == [1, 2, 2]
    assert norm_stats["stride"].tolist() == [1, 1, 1]
    assert norm_stats["output_stride"].tolist() == [1, 4, 4]
    assert norm_stats["lost"] == second_stats["lost"]
    assert streaming.output_stride.tolist() == [1, 4, 4]

    reference_input = torch.randn(
        batch, in_channels, *image_hw, dtype=torch.double, requires_grad=True
    )
    full_input = reference_input.detach().clone().requires_grad_(True)
    streaming_input = reference_input.detach().clone().requires_grad_(True)

    reference_output = reference(reference_input).permute(0, 3, 1, 2)
    full_output = full_nchw(full_input)
    streaming_output = streaming(streaming_input)
    assert not streaming_output.requires_grad
    expected_shape = (batch, embed_dim, 8, 10)
    assert reference_output.shape == full_output.shape == streaming_output.shape
    assert reference_output.shape == expected_shape
    torch.testing.assert_close(full_output, reference_output)
    torch.testing.assert_close(streaming_output, reference_output)

    upstream = torch.randn(expected_shape, dtype=torch.double)
    reference_output.backward(upstream)
    full_output.backward(upstream)
    streaming.backward(streaming_input, upstream)
    torch.testing.assert_close(full_input.grad, reference_input.grad)
    torch.testing.assert_close(streaming_input.grad, reference_input.grad)

    parameter_groups = (
        list(reference.parameters()),
        list(full_nchw.parameters()),
        list(streaming.stream_module.parameters()),
    )
    for reference_parameter, full_parameter, streaming_parameter in zip(
        *parameter_groups
    ):
        assert reference_parameter.requires_grad == full_parameter.requires_grad
        assert reference_parameter.requires_grad == streaming_parameter.requires_grad
        if reference_parameter.requires_grad:
            torch.testing.assert_close(full_parameter.grad, reference_parameter.grad)
            torch.testing.assert_close(
                streaming_parameter.grad, reference_parameter.grad
            )

    optimizers = [
        torch.optim.SGD(parameters, lr=0.025) for parameters in parameter_groups
    ]
    for optimizer in optimizers:
        optimizer.step()
    for reference_parameter, full_parameter, streaming_parameter in zip(
        *parameter_groups
    ):
        torch.testing.assert_close(full_parameter, reference_parameter)
        torch.testing.assert_close(streaming_parameter, reference_parameter)


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


class _LinearNHWCBackend(nn.Module):
    """Exercise autocast-sensitive matrix multiplication in replay."""

    def __init__(self, channels=4):
        super().__init__()
        self.kernel_size = 3
        self.dilation = 1
        self.projection = nn.Linear(channels, channels)

    def forward(self, value):
        return self.projection(value)


@pytest.mark.cuda_integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_streaming_attention_backward_restores_cuda_autocast_state():
    attention = StreamingNeighborhoodAttention2D(
        attention=_LinearNHWCBackend()
    ).cuda()
    attention.input_loc = Box(0, 0, 0, 0, None)
    input = torch.randn(1, 4, 5, 5, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = attention(input)
        assert output.dtype == torch.bfloat16

    # Backward deliberately starts after the user autocast context has ended.
    # custom_bwd must restore the forward state while constructing the replay
    # graph, otherwise Linear backward mixes BFloat16 and Float matrices.
    output.sum().backward()

    assert input.grad is not None
    assert attention.attention.projection.weight.grad is not None


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

    def __init__(self, attention, channels, hidden_channels, layer_scale=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attn = attention
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = _LinearMlp(channels, hidden_channels)
        if layer_scale is not None:
            self.gamma1 = nn.Parameter(torch.full((channels,), layer_scale))
            self.gamma2 = nn.Parameter(torch.full((channels,), layer_scale))

    def forward(self, x):
        attention = self.attn(self.norm1(x))
        if hasattr(self, "gamma1"):
            attention = self.gamma1 * attention
        x = x + attention
        mlp = self.mlp(self.norm2(x))
        if hasattr(self, "gamma2"):
            mlp = self.gamma2 * mlp
        return x + mlp


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
    for gamma_name in ("gamma1", "gamma2"):
        if hasattr(reference, gamma_name):
            yield (
                gamma_name,
                getattr(reference, gamma_name),
                getattr(nchw, gamma_name).weight,
            )


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
    """Restore NCHW pointwise parameters to their original NAT shapes."""

    if value.ndim == reference.ndim + 2:
        return value[:, :, 0, 0]
    if value.shape != reference.shape and value.numel() == reference.numel():
        return value.reshape_as(reference)
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
        assert attention_parameters[_REL_POS_BIAS_PARAMETER].grad is not None, (
            f"{execution} relative-position-bias gradient is missing"
        )
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
        assert full_parameter.grad is not None, (
            f"full-frame gradient missing for {name!r}"
        )
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
        layer_scale=1e-5,
    ).float()
    nchw_source = NCHWNATLayer(
        _make_natten(natten_backend, channels, heads, kernel_size, dilation=1),
        channels,
        hidden_channels,
        layer_scale=1e-5,
    ).float()
    copy_nhwc_nat_to_nchw(reference, nchw_source)
    full_nchw = copy.deepcopy(nchw_source)
    streaming = StreamingCNN(
        nchw_source,
        tile_shape=(batch, channels, *tile_shape),
        copy_to_gpu=True,
    )
    _assert_identical_state_dict(full_nchw, streaming)

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
    assert isinstance(streaming.stream_module.gamma1, StreamingLayerScale)
    assert isinstance(streaming.stream_module.gamma2, StreamingLayerScale)

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

        _assert_identical_state_dict(full_nchw, streaming)
        reference_output = reference(reference_input)
        full_output = full_nchw(nchw_input)
        streaming_output = streaming(streaming_input)
        torch.testing.assert_close(
            full_output,
            reference_output.permute(0, 3, 1, 2),
            rtol=_CROSS_LAYOUT_OUTPUT_RTOL,
            atol=_CROSS_LAYOUT_OUTPUT_ATOL,
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
            rtol=_CROSS_LAYOUT_INPUT_GRAD_RTOL,
            atol=_CROSS_LAYOUT_INPUT_GRAD_ATOL,
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
            _assert_close_with_diagnostics(
                _linear_shaped(full_parameter.grad, reference_parameter.grad),
                reference_parameter.grad,
                rtol=_CROSS_LAYOUT_PARAMETER_GRAD_RTOL,
                atol=_CROSS_LAYOUT_PARAMETER_GRAD_ATOL,
                quantity="cross-layout parameter gradient",
                parameter_name=name,
                cycle=cycle,
            )
        for (name, _, full_parameter), (streaming_name, _, streaming_parameter) in zip(
            full_pairs, streaming_pairs
        ):
            assert name == streaming_name
            assert streaming_parameter.grad is not None, (
                f"missing NCHW gradient for {name}"
            )
            _assert_close_with_diagnostics(
                streaming_parameter.grad,
                full_parameter.grad,
                rtol=_SAME_LAYOUT_RTOL,
                atol=_SAME_LAYOUT_ATOL,
                quantity="same-layout streaming parameter gradient",
                parameter_name=name,
                cycle=cycle,
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
                    rtol=_CROSS_LAYOUT_OUTPUT_RTOL,
                    atol=_CROSS_LAYOUT_OUTPUT_ATOL,
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

            # Independently accumulated gradients may introduce small numerical
            # drift. Restore both NCHW paths before the reset/reuse cycle.
            copy_nhwc_nat_to_nchw(reference, full_nchw)
            copy_nhwc_nat_to_nchw(reference, streaming.stream_module)
            _assert_identical_state_dict(full_nchw, streaming)


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
        layer_scale=1e-5,
    ).float()
    nchw_source = NCHWNATBlock(
        channels=channels,
        depth=len(dilations),
        num_heads=heads,
        kernel_size=kernel_size,
        dilations=dilations,
        downsample=False,
        mlp_ratio=mlp_ratio,
        layer_scale=1e-5,
    ).float()
    copy_nhwc_nat_block_to_nchw(reference, nchw_source)
    full_nchw = copy.deepcopy(nchw_source)
    streaming = StreamingCNN(
        nchw_source, tile_shape=(batch, channels, *tile_hw), copy_to_gpu=True
    )
    _assert_identical_state_dict(full_nchw, streaming)

    streamed_attentions = [layer.attn for layer in streaming.stream_module.blocks]
    assert all(
        isinstance(item, StreamingNeighborhoodAttention2D)
        for item in streamed_attentions
    )
    assert len({id(item.seen_indices) for item in streamed_attentions}) == len(
        dilations
    )
    assert reference.downsample is None and full_nchw.downsample is None
    assert all(
        isinstance(layer.gamma1, StreamingLayerScale)
        and isinstance(layer.gamma2, StreamingLayerScale)
        for layer in streaming.stream_module.blocks
    )

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

        _assert_identical_state_dict(full_nchw, streaming)
        reference_output = reference(reference_input)
        full_output = full_nchw(full_input)
        streaming_output = streaming(streaming_input)
        expected_output = reference_output.permute(0, 3, 1, 2)
        _assert_close_with_diagnostics(
            full_output,
            expected_output,
            rtol=_CROSS_LAYOUT_OUTPUT_RTOL,
            atol=_CROSS_LAYOUT_OUTPUT_ATOL,
            quantity="cross-layout output",
            cycle=cycle,
        )
        _assert_close_with_diagnostics(
            streaming_output,
            full_output,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            quantity="same-layout streaming output",
            cycle=cycle,
        )

        reference_output.backward(upstream)
        nchw_upstream = upstream.permute(0, 3, 1, 2).contiguous()
        full_output.backward(nchw_upstream)
        streaming.backward(streaming_input, nchw_upstream)
        expected_input_grad = reference_input.grad.permute(0, 3, 1, 2)
        _assert_close_with_diagnostics(
            full_input.grad,
            expected_input_grad,
            rtol=_CROSS_LAYOUT_INPUT_GRAD_RTOL,
            atol=_CROSS_LAYOUT_INPUT_GRAD_ATOL,
            quantity="cross-layout input gradient",
            cycle=cycle,
        )
        _assert_close_with_diagnostics(
            streaming_input.grad,
            full_input.grad,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            quantity="same-layout streaming input gradient",
            cycle=cycle,
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
                _assert_close_with_diagnostics(
                    _linear_shaped(full_parameter.grad, reference_parameter.grad),
                    reference_parameter.grad,
                    rtol=_CROSS_LAYOUT_PARAMETER_GRAD_RTOL,
                    atol=_CROSS_LAYOUT_PARAMETER_GRAD_ATOL,
                    quantity="cross-layout accumulated parameter gradient",
                    parameter_name=f"blocks.{layer_index}.{name}",
                    cycle=cycle,
                )
            for full_pair, streaming_pair in zip(full_pairs, streaming_pairs):
                name, _, full_parameter = full_pair
                streaming_name, _, streaming_parameter = streaming_pair
                assert name == streaming_name
                assert streaming_parameter.grad is not None
                _assert_close_with_diagnostics(
                    streaming_parameter.grad,
                    full_parameter.grad,
                    rtol=_SAME_LAYOUT_RTOL,
                    atol=_SAME_LAYOUT_ATOL,
                    quantity="same-layout streaming parameter gradient",
                    parameter_name=f"blocks.{layer_index}.{name}",
                    cycle=cycle,
                )

        for attention in streamed_attentions:
            assert attention.input_loc is None
            assert (attention.seen_indices.y, attention.seen_indices.height) == (0, 0)
            assert (attention.seen_indices.x, attention.seen_indices.width) == (0, 0)
            assert attention.seen_indices.sides is None

        # Cycle 0 alone exercises optimizer parity.  The independently computed
        # cross-layout gradients are only approximately equal, so the three
        # parameter sets are expected to differ slightly after their steps.
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
                        rtol=_CROSS_LAYOUT_OUTPUT_RTOL,
                        atol=_CROSS_LAYOUT_OUTPUT_ATOL,
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

            # Make the updated NHWC reference canonical before exercising a
            # second forward/backward pass.  This keeps cycle 1 focused on
            # reuse/reset of this exact StreamingCNN rather than compounding
            # the approximate cross-layout gradients from cycle 0.
            copy_nhwc_nat_block_to_nchw(reference, full_nchw)
            copy_nhwc_nat_block_to_nchw(reference, streaming.stream_module)
            _assert_identical_state_dict(full_nchw, streaming)

            for optimizer in (reference_optimizer, full_optimizer, streaming_optimizer):
                optimizer.zero_grad(set_to_none=True)


def test_complete_four_stage_nat_matches_nhwc_full_and_cached_streaming(
    natten_backend,
):
    """The complete backbone preserves training and streaming state parity."""

    from lightstream.models.nat.nat import NAT

    torch.manual_seed(42001)
    configuration = dict(
        embed_dim=8,
        mlp_ratio=2,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8],
        drop_path_rate=0.0,
        kernel_size=3,
        num_classes=0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
    )
    reference = NAT(**configuration).float()
    nchw_configuration = {
        key: value
        for key, value in configuration.items()
        if key not in {"num_classes", "layer_scale"}
    }
    full_nchw = NCHWNAT(**nchw_configuration).float()
    converted_state = convert_nhwc_nat_state_dict(reference.state_dict())
    load_result = full_nchw.load_state_dict(converted_state, strict=False)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert set(converted_state) == set(full_nchw.state_dict())

    # A single full-image tile keeps this test focused on complete-model
    # conversion, forward/backward parity, optimizer parity, stream reset, and
    # tile-cache reconstruction. Multi-tile seam behavior is covered by the
    # focused attention, block, and downsampler tests: a true multi-tile
    # four-stage test would need a tile of roughly 193 pixels per side to leave
    # a valid interior after the model's approximately 183-pixel receptive
    # field.
    tile_shape = (1, 3, 97, 99)
    streaming = StreamingCNN(
        copy.deepcopy(full_nchw), tile_shape=tile_shape, copy_to_gpu=True
    )
    tile_cache = streaming.get_tile_cache()

    def parameter_pairs(nchw_module):
        reference_parameters = dict(reference.named_parameters())
        nchw_parameters = dict(nchw_module.named_parameters())
        converted_names = {
            next(iter(convert_nhwc_nat_state_dict({name: parameter}))): name
            for name, parameter in reference_parameters.items()
        }
        assert set(converted_names) == set(nchw_parameters)
        for nchw_name, reference_name in converted_names.items():
            yield (
                nchw_name,
                reference_parameters[reference_name],
                nchw_parameters[nchw_name],
            )

    def compare_cycle(streamers, seed):
        for module in (
            reference,
            full_nchw,
            *(item.stream_module for item in streamers),
        ):
            module.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        reference_input = torch.randn(
            1, 3, 97, 99, dtype=torch.float32, requires_grad=True
        )
        full_input = reference_input.detach().clone().requires_grad_(True)
        stream_inputs = [
            full_input.detach().clone().requires_grad_(True) for _ in streamers
        ]

        reference_features = reference.forward_feature_map(reference_input)
        full_features = full_nchw(full_input)
        stream_features = [item(value) for item, value in zip(streamers, stream_inputs)]
        expected_features = reference_features.permute(0, 3, 1, 2)
        _assert_close_with_diagnostics(
            full_features,
            expected_features,
            rtol=_CROSS_LAYOUT_OUTPUT_RTOL,
            atol=_CROSS_LAYOUT_OUTPUT_ATOL,
            quantity="complete four-stage features",
        )
        for output in stream_features:
            torch.testing.assert_close(
                output, full_features, rtol=_SAME_LAYOUT_RTOL, atol=_SAME_LAYOUT_ATOL
            )

        torch.manual_seed(seed + 1)
        upstream = torch.randn_like(full_features)
        reference_features.backward(upstream.permute(0, 2, 3, 1).contiguous())
        full_features.backward(upstream)
        for item, value in zip(streamers, stream_inputs):
            item.backward(value, upstream)
        _assert_close_with_diagnostics(
            full_input.grad,
            reference_input.grad,
            rtol=_CROSS_LAYOUT_INPUT_GRAD_RTOL,
            atol=_CROSS_LAYOUT_INPUT_GRAD_ATOL,
            quantity="complete four-stage image gradient",
        )
        for value in stream_inputs:
            torch.testing.assert_close(
                value.grad,
                full_input.grad,
                rtol=_SAME_LAYOUT_RTOL,
                atol=_SAME_LAYOUT_ATOL,
            )

        full_pairs = list(parameter_pairs(full_nchw))
        for name, reference_parameter, nchw_parameter in full_pairs:
            assert (
                reference_parameter.grad is not None and nchw_parameter.grad is not None
            )
            _assert_close_with_diagnostics(
                _linear_shaped(nchw_parameter.grad, reference_parameter.grad),
                reference_parameter.grad,
                rtol=_CROSS_LAYOUT_PARAMETER_GRAD_RTOL,
                atol=_CROSS_LAYOUT_PARAMETER_GRAD_ATOL,
                quantity="complete four-stage parameter gradient",
                parameter_name=name,
            )
        for item in streamers:
            for (name, _, full_parameter), (_, _, stream_parameter) in zip(
                full_pairs, parameter_pairs(item.stream_module)
            ):
                torch.testing.assert_close(
                    stream_parameter.grad,
                    full_parameter.grad,
                    rtol=_SAME_LAYOUT_RTOL,
                    atol=_SAME_LAYOUT_ATOL,
                    msg=f"streamed parameter gradient {name}",
                )

    compare_cycle([streaming], 43001)
    optimizers = [
        torch.optim.SGD(module.parameters(), lr=0.01)
        for module in (reference, full_nchw, streaming.stream_module)
    ]
    for optimizer in optimizers:
        optimizer.step()
    for name, reference_parameter, full_parameter in parameter_pairs(full_nchw):
        torch.testing.assert_close(
            _linear_shaped(full_parameter, reference_parameter),
            reference_parameter,
            rtol=_CROSS_LAYOUT_OUTPUT_RTOL,
            atol=_CROSS_LAYOUT_OUTPUT_ATOL,
            msg=f"optimizer-updated parameter {name}",
        )
    for (name, _, full_parameter), (_, _, stream_parameter) in zip(
        parameter_pairs(full_nchw), parameter_pairs(streaming.stream_module)
    ):
        torch.testing.assert_close(
            stream_parameter,
            full_parameter,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            msg=f"streamed optimizer-updated parameter {name}",
        )

    # Make one updated NCHW state canonical, then verify both the reset stream
    # and a separately constructed stream restored from the saved cache.
    streaming.stream_module.load_state_dict(full_nchw.state_dict())
    cached_streaming = StreamingCNN(
        copy.deepcopy(full_nchw),
        tile_shape=tile_shape,
        copy_to_gpu=True,
        state_dict=tile_cache,
    )
    cached_streaming.stream_module.load_state_dict(full_nchw.state_dict())
    compare_cycle([streaming, cached_streaming], 44001)


@pytest.mark.cuda_integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_complete_four_stage_nat_multi_tile_cuda_parity(natten_backend):
    """A complete reduced NAT is exact across shifted, multi-tile streaming."""

    from lightstream.models.nat.nat import NAT

    torch.manual_seed(45001)
    configuration = dict(
        embed_dim=8,
        mlp_ratio=2,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8],
        drop_path_rate=0.0,
        kernel_size=3,
        num_classes=0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
    )
    reference = NAT(**configuration).float().cuda()
    nchw_configuration = {
        key: value
        for key, value in configuration.items()
        if key not in {"num_classes", "layer_scale"}
    }
    full_nchw = NCHWNAT(**nchw_configuration).float().cuda()
    full_nchw.load_state_dict(convert_nhwc_nat_state_dict(reference.state_dict()))

    # The approximately 183-pixel value is the model's forward receptive field,
    # while backward-safe streaming statistics require the larger 385×389
    # physical tile used here.
    tile_shape = (1, 3, 385, 389)
    image_shape = (481, 485)
    streaming = StreamingCNN(copy.deepcopy(full_nchw), tile_shape=tile_shape)
    tile_cache = streaming.get_tile_cache()

    # Establish the intended traversal before starting either expensive parity
    # cycle. In particular, guard against receptive-field changes shrinking the
    # valid step or quietly turning this back into a regular-grid test.
    valid_heights, valid_widths = streaming._compute_valid_output_sizes()
    valid_step_height, valid_step_width = streaming._compute_valid_input_step(
        valid_heights, valid_widths
    )
    align_height, align_width = (
        int(value) for value in streaming._compute_internal_alignment()
    )
    assert (align_height, align_width) == (32, 32)
    assert (valid_step_height, valid_step_width) == (64, 64), (
        streaming.tile_gradient_lost,
        streaming._compute_internal_safe_input_step(),
        streaming._tile_output_shapes[0],
        streaming._tile_output_lost[0],
    )
    assert image_shape[0] % 2 == 1
    assert image_shape[1] % 2 == 1
    assert image_shape[0] != image_shape[1]
    assert all(image > tile for image, tile in zip(image_shape, tile_shape[-2:]))
    n_rows, n_cols = streaming._compute_tile_grid(
        *image_shape,
        tile_shape[-2],
        tile_shape[-1],
        valid_step_height,
        valid_step_width,
    )
    assert (n_rows, n_cols) == (3, 3)
    shape_only_image = torch.empty(1, 3, *image_shape, device="meta")
    expected_starts = [
        (y, x)
        for y, x, _ in streaming._iter_input_tiles(
            shape_only_image,
            n_rows,
            n_cols,
            valid_step_height,
            valid_step_width,
            tile_shape[-2],
            tile_shape[-1],
        )
    ]
    rows = sorted({y for y, _ in expected_starts})
    columns = sorted({x for _, x in expected_starts})
    assert rows == [0, 64, 96]
    assert columns == [0, 64, 96]
    full_heights, full_widths = streaming._compute_full_output_sizes(shape_only_image)
    assert full_heights[0] > valid_heights[0]
    assert full_widths[0] > valid_widths[0]

    net_stats = tile_cache["net_stats"]
    attention_names = [name for name in net_stats if name.endswith(".attn")]
    downsampler_names = [
        name for name in net_stats if name.endswith(".downsample.norm")
    ]
    boundary_names = [
        "patch_embed.norm",
        *attention_names,
        *downsampler_names,
        "norm",
    ]
    boundary_stats = {name: net_stats[name] for name in boundary_names}
    for name, stats in boundary_stats.items():
        output_height, output_width = stats["output_shape"][-2:]
        lost = stats["lost"]
        assert output_height - lost.top - lost.bottom > 0, name
        assert output_width - lost.left - lost.right > 0, name

    def parameter_pairs(nchw_module):
        reference_parameters = dict(reference.named_parameters())
        nchw_parameters = dict(nchw_module.named_parameters())
        converted_names = {
            next(iter(convert_nhwc_nat_state_dict({name: parameter}))): name
            for name, parameter in reference_parameters.items()
        }
        assert set(converted_names) == set(nchw_parameters)
        for nchw_name, reference_name in converted_names.items():
            yield (
                nchw_name,
                reference_parameters[reference_name],
                nchw_parameters[nchw_name],
            )

    def compare_cycle(streamers, *, phase, seed):
        modules = (
            reference,
            full_nchw,
            *(item.stream_module for _, item in streamers),
        )
        for module in modules:
            module.zero_grad(set_to_none=True)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        reference_input = torch.randn(
            1, 3, *image_shape, generator=generator, device="cuda", requires_grad=True
        )
        full_input = reference_input.detach().clone().requires_grad_(True)
        stream_inputs = [
            full_input.detach().clone().requires_grad_(True) for _ in streamers
        ]

        reference_features = reference.forward_feature_map(reference_input)
        full_features = full_nchw(full_input)
        stream_features = [
            item(value) for (_, item), value in zip(streamers, stream_inputs)
        ]
        expected_features = reference_features.permute(0, 3, 1, 2)
        _assert_close_with_diagnostics(
            full_features,
            expected_features,
            rtol=_COMPLETE_MODEL_FEATURE_RTOL,
            atol=_COMPLETE_MODEL_FEATURE_ATOL,
            quantity="multi-tile complete four-stage features",
            cycle=phase,
        )
        for (streamer_name, item), output in zip(streamers, stream_features):
            _assert_close_with_diagnostics(
                output,
                full_features,
                rtol=_COMPLETE_MODEL_STREAMED_FEATURE_RTOL,
                atol=_COMPLETE_MODEL_STREAMED_FEATURE_ATOL,
                quantity="multi-tile complete four-stage streamed feature map",
                cycle=f"{phase}, {streamer_name}",
                streaming=item,
            )

        # Verify the actual forward context, rather than merely relying on the
        # selected image dimensions: both dimensions traverse multiple tiles,
        # and neither final tile lies on the regular stepping grid.
        for (_, item), output in zip(streamers, stream_features):
            starts = [(y, x) for y, x, _ in item._last_forward_tiles]
            assert starts == expected_starts
            rows = sorted({y for y, _ in starts})
            columns = sorted({x for _, x in starts})
            assert len(rows) >= 2 and len(columns) >= 2
            valid_heights, valid_widths = item._compute_valid_output_sizes()
            valid_step_height, valid_step_width = item._compute_valid_input_step(
                valid_heights, valid_widths
            )
            assert output.shape[-2] > valid_heights[0]
            assert output.shape[-1] > valid_widths[0]
            assert rows[-1] != (len(rows) - 1) * valid_step_height
            assert columns[-1] != (len(columns) - 1) * valid_step_width

            regular_rows = rows[:-1]
            regular_columns = columns[:-1]
            shifted_row = rows[-1]
            shifted_column = columns[-1]
            strided_boundary_names = ["patch_embed.norm", *downsampler_names]
            for name in strided_boundary_names:
                stats = boundary_stats[name]
                stride_y, stride_x = (
                    int(value) for value in stats["output_stride"][-2:]
                )
                assert all(row % stride_y == 0 for row in regular_rows), name
                assert all(column % stride_x == 0 for column in regular_columns), name
                assert shifted_row % stride_y == 0, name
                assert shifted_column % stride_x == 0, name

            final_stride_y, final_stride_x = (
                int(value) for value in boundary_stats["norm"]["output_stride"][-2:]
            )
            assert (shifted_row - regular_rows[-1]) // final_stride_y >= 1
            assert (shifted_column - regular_columns[-1]) // final_stride_x >= 1

        upstream = torch.randn(full_features.shape, generator=generator, device="cuda")
        reference_features.backward(upstream.permute(0, 2, 3, 1).contiguous())
        full_features.backward(upstream)
        for (_, item), value in zip(streamers, stream_inputs):
            item.backward(value, upstream)

        assert reference_input.grad is not None
        assert full_input.grad is not None
        _assert_close_with_diagnostics(
            full_input.grad,
            reference_input.grad,
            rtol=_COMPLETE_MODEL_IMAGE_GRAD_RTOL,
            atol=_COMPLETE_MODEL_IMAGE_GRAD_ATOL,
            quantity="multi-tile complete four-stage image gradient",
            cycle=phase,
        )
        for (streamer_name, item), value in zip(streamers, stream_inputs):
            assert value.grad is not None
            _assert_close_with_diagnostics(
                value.grad,
                full_input.grad,
                rtol=_COMPLETE_MODEL_STREAMED_IMAGE_GRAD_RTOL,
                atol=_COMPLETE_MODEL_STREAMED_IMAGE_GRAD_ATOL,
                quantity="multi-tile complete four-stage streamed image gradient",
                cycle=f"{phase}, {streamer_name}",
                streaming=item,
            )

        full_pairs = list(parameter_pairs(full_nchw))
        for name, reference_parameter, full_parameter in full_pairs:
            assert (
                reference_parameter.grad is not None and full_parameter.grad is not None
            )
            _assert_close_with_diagnostics(
                _linear_shaped(full_parameter.grad, reference_parameter.grad),
                reference_parameter.grad,
                rtol=_COMPLETE_MODEL_PARAMETER_GRAD_RTOL,
                atol=_COMPLETE_MODEL_PARAMETER_GRAD_ATOL,
                quantity="multi-tile complete four-stage parameter gradient",
                parameter_name=name,
                cycle=phase,
            )
        for streamer_name, item in streamers:
            streamed_pairs = list(parameter_pairs(item.stream_module))
            assert len(streamed_pairs) == len(full_pairs)
            for (name, _, full_parameter), (
                stream_name,
                _,
                stream_parameter,
            ) in zip(full_pairs, streamed_pairs):
                assert stream_name == name
                assert full_parameter.grad is not None
                assert stream_parameter.grad is not None
                _assert_close_with_diagnostics(
                    stream_parameter.grad,
                    full_parameter.grad,
                    rtol=_COMPLETE_MODEL_STREAMED_PARAMETER_GRAD_RTOL,
                    atol=_COMPLETE_MODEL_STREAMED_PARAMETER_GRAD_ATOL,
                    quantity=(
                        "multi-tile complete four-stage streamed parameter gradient"
                    ),
                    parameter_name=name,
                    cycle=f"{phase}, {streamer_name}",
                )

    initial_phase = "initial cycle"
    compare_cycle([("original streamer", streaming)], phase=initial_phase, seed=46001)
    optimizers = [
        torch.optim.SGD(module.parameters(), lr=0.01)
        for module in (reference, full_nchw, streaming.stream_module)
    ]
    for optimizer in optimizers:
        optimizer.step()
    for name, reference_parameter, full_parameter in parameter_pairs(full_nchw):
        _assert_close_with_diagnostics(
            _linear_shaped(full_parameter, reference_parameter),
            reference_parameter,
            rtol=_COMPLETE_MODEL_PARAMETER_RTOL,
            atol=_COMPLETE_MODEL_PARAMETER_ATOL,
            quantity="multi-tile optimizer-updated parameter",
            parameter_name=name,
            cycle=initial_phase,
        )
    for (name, _, full_parameter), (stream_name, _, stream_parameter) in zip(
        parameter_pairs(full_nchw), parameter_pairs(streaming.stream_module)
    ):
        assert stream_name == name
        torch.testing.assert_close(
            stream_parameter,
            full_parameter,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            msg=f"multi-tile streamed optimizer-updated parameter {name}",
        )

    streaming.stream_module.load_state_dict(full_nchw.state_dict())
    cached_streaming = StreamingCNN(
        copy.deepcopy(full_nchw), tile_shape=tile_shape, state_dict=tile_cache
    )
    cached_streaming.stream_module.load_state_dict(full_nchw.state_dict())
    compare_cycle(
        [
            ("original streamer", streaming),
            ("cache-restored streamer", cached_streaming),
        ],
        phase="post-optimizer cycle",
        seed=47001,
    )


def test_first_nat_stage_shifted_tiles_preserve_downsampler_phase(natten_backend):
    """The first complete stage stays exact at the shifted stride-2 boundary.

    This uses the physical tile and image sizes from the four-stage integration
    test.  Keeping the tokenizer in the prefix is important: the downsampler's
    stride is cumulative with the tokenization stride, rather than relative to
    the stage input.
    """

    torch.manual_seed(45001)
    model = NCHWNAT(
        embed_dim=8,
        mlp_ratio=2,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8],
        drop_path_rate=0.0,
        kernel_size=3,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
    ).float()
    prefix = nn.Sequential(
        copy.deepcopy(model.patch_embed), copy.deepcopy(model.levels[0])
    )
    full_prefix = copy.deepcopy(prefix)
    tile_shape = (1, 3, 257, 261)
    image_shape = (339, 351)
    streaming = StreamingCNN(prefix, tile_shape=tile_shape, copy_to_gpu=True)

    value = torch.randn(1, 3, *image_shape)
    expected = full_prefix(value)
    actual = streaming(value)
    starts = [(y, x) for y, x, _ in streaming._last_forward_tiles]
    rows = sorted({y for y, _ in starts})
    columns = sorted({x for _, x in starts})
    valid_heights, valid_widths = streaming._compute_valid_output_sizes()
    step_y, step_x = streaming._compute_valid_input_step(valid_heights, valid_widths)

    assert len(rows) >= 2 and len(columns) >= 2
    assert rows[-1] != (len(rows) - 1) * step_y
    assert columns[-1] != (len(columns) - 1) * step_x
    # Tokenizer stride 4 followed by the stage downsampler stride 2.
    assert streaming.output_stride.tolist() == [1, 8, 8]
    assert all(y % 8 == 0 and x % 8 == 0 for y, x in starts)
    _assert_close_with_diagnostics(
        actual,
        expected,
        rtol=_SAME_LAYOUT_RTOL,
        atol=_SAME_LAYOUT_ATOL,
        quantity="first NAT stage shifted-tile downsampler output",
        streaming=streaming,
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
    _assert_identical_state_dict(full_nchw, streaming)

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

    _assert_identical_state_dict(full_nchw, streaming)
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
        rtol=_CROSS_LAYOUT_OUTPUT_RTOL,
        atol=_CROSS_LAYOUT_OUTPUT_ATOL,
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
    _assert_close_with_diagnostics(
        full_input.grad,
        expected_input_grad,
        rtol=_CROSS_LAYOUT_INPUT_GRAD_RTOL,
        atol=_CROSS_LAYOUT_INPUT_GRAD_ATOL,
        quantity="cross-layout downsampler input gradient",
        cycle=0,
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
    explicitly_required_names = {
        "downsample.reduction.weight",
        "downsample.norm.weight",
        "downsample.norm.bias",
    }
    for layer_index in range(len(reference.blocks)):
        explicitly_required_names.update(
            {
                f"blocks.{layer_index}.norm1.weight",
                f"blocks.{layer_index}.norm1.bias",
                f"blocks.{layer_index}.norm2.weight",
                f"blocks.{layer_index}.norm2.bias",
                f"blocks.{layer_index}.attn.qkv.weight",
                f"blocks.{layer_index}.attn.proj.weight",
                f"blocks.{layer_index}.attn.{_REL_POS_BIAS_PARAMETER}",
                f"blocks.{layer_index}.mlp.fc1.weight",
                f"blocks.{layer_index}.mlp.fc2.weight",
            }
        )
    assert explicitly_required_names <= reference_names
    for name, reference_parameter, full_parameter in full_pairs:
        assert reference_parameter.grad is not None
        assert full_parameter.grad is not None
        _assert_close_with_diagnostics(
            _linear_shaped(full_parameter.grad, reference_parameter.grad),
            reference_parameter.grad,
            rtol=_CROSS_LAYOUT_PARAMETER_GRAD_RTOL,
            atol=_CROSS_LAYOUT_PARAMETER_GRAD_ATOL,
            quantity="cross-layout accumulated parameter gradient",
            parameter_name=name,
            cycle=0,
        )
    for full_pair, streaming_pair in zip(full_pairs, streaming_pairs):
        name, _, full_parameter = full_pair
        streaming_name, _, streaming_parameter = streaming_pair
        assert name == streaming_name
        assert streaming_parameter.grad is not None
        _assert_close_with_diagnostics(
            streaming_parameter.grad,
            full_parameter.grad,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            quantity="same-layout streaming parameter gradient",
            parameter_name=name,
            cycle=0,
        )

    # Optimizer validation is deliberately terminal: each independently
    # accumulated gradient is applied exactly once, and those diverged states
    # must not feed another parity cycle.  Retain a copy of every parameter so
    # the update itself can be compared in addition to its resulting value.
    reference_before_step = {
        name: parameter.detach().clone() for name, parameter, _ in full_pairs
    }
    full_before_step = {
        name: parameter.detach().clone() for name, _, parameter in full_pairs
    }
    streaming_before_step = {
        name: parameter.detach().clone() for name, _, parameter in streaming_pairs
    }
    assert len(reference_before_step) == len(dict(reference.named_parameters()))
    assert len(full_before_step) == len(dict(full_nchw.named_parameters()))
    assert len(streaming_before_step) == len(
        dict(streaming.stream_module.named_parameters())
    )

    reference_optimizer, full_optimizer, streaming_optimizer = optimizers
    reference_optimizer.step()
    full_optimizer.step()
    streaming_optimizer.step()

    full_pairs = list(_nat_block_parameter_pairs(reference, full_nchw))
    streaming_pairs = list(
        _nat_block_parameter_pairs(reference, streaming.stream_module)
    )
    for name, reference_parameter, full_parameter in full_pairs:
        _assert_close_with_diagnostics(
            _linear_shaped(full_parameter, reference_parameter),
            reference_parameter,
            rtol=_CROSS_LAYOUT_OUTPUT_RTOL,
            atol=_CROSS_LAYOUT_OUTPUT_ATOL,
            quantity="cross-layout optimizer-updated parameter value",
            parameter_name=name,
            cycle=0,
        )
        reference_update = reference_parameter - reference_before_step[name]
        full_update = full_parameter - full_before_step[name]
        _assert_close_with_diagnostics(
            _linear_shaped(full_update, reference_update),
            reference_update,
            rtol=_CROSS_LAYOUT_PARAMETER_GRAD_RTOL,
            atol=0.015 * _CROSS_LAYOUT_PARAMETER_GRAD_ATOL,
            quantity="cross-layout optimizer update delta",
            parameter_name=name,
            cycle=0,
        )
    for full_pair, streaming_pair in zip(full_pairs, streaming_pairs):
        name, _, full_parameter = full_pair
        streaming_name, _, streaming_parameter = streaming_pair
        assert name == streaming_name
        _assert_close_with_diagnostics(
            streaming_parameter,
            full_parameter,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            quantity="same-layout optimizer-updated parameter value",
            parameter_name=name,
            cycle=0,
        )
        full_update = full_parameter - full_before_step[name]
        streaming_update = streaming_parameter - streaming_before_step[name]
        _assert_close_with_diagnostics(
            streaming_update,
            full_update,
            rtol=_SAME_LAYOUT_RTOL,
            atol=_SAME_LAYOUT_ATOL,
            quantity="same-layout optimizer update delta",
            parameter_name=name,
            cycle=0,
        )
