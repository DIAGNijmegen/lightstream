import logging
import math

import torch
import torch.nn as nn
import pytest

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.layers import (
    ChannelLayerNorm,
    LayerScale,
    StatisticsProbe,
    StreamingChannelLayerNorm,
    StreamingConv2d,
    StreamingLayerScale,
    StreamingMerge,
    StreamingUpsample2d,
)
from lightstream.core.scnn.scnn import StreamingCNN, _resize_nearest_bool_mask
from lightstream.core.scnn.utils import Box, Lost, Sides
from lightstream.models.testnet.segment import StreamingTestNet
from lightstream.models.testnet.testnet import StreamingTestNet as SaliencyTestNet

from lightstream.core.reducer import (
    BaseReducer,
    BaseStreamingGlobalReducer,
    AttentionGeMReducer,
    FusedAttentionGeMReducer,
    GeMReducer,
    MeanReducer,
    NGWPReducer,
    StreamingAttentionGeMReducer,
    StreamingFusedAttentionGeMReducer,
    StreamingGeMReducer,
    StreamingMeanReducer,
    StreamingNGWPReducer,
    StreamingSumReducer,
    SumReducer,
)


@pytest.mark.parametrize(
    ("tile_hw", "image_hw"),
    [
        ((8, 8), (17, 17)),  # even setup tile, odd full image
        ((9, 9), (17, 17)),  # odd setup tile, odd full image
        ((8, 9), (17, 19)),  # independent height/width phases
    ],
)
def test_strided_conv_full_output_shape_and_tile_placement_preserve_phase(tile_hw, image_hw):
    torch.manual_seed(91)
    reference = nn.Conv2d(2, 3, kernel_size=3, stride=2, padding=1).eval()
    streaming_module = nn.Conv2d(2, 3, kernel_size=3, stride=2, padding=1).eval()
    streaming_module.load_state_dict(reference.state_dict())
    streaming = StreamingCNN(
        streaming_module,
        tile_shape=(1, 2, *tile_hw),
        deterministic=True,
        copy_to_gpu=False,
    )
    image = torch.randn(1, 2, *image_hw)

    expected = reference(image)
    actual = streaming(image)

    assert actual.shape == expected.shape == (1, 3, math.ceil(image_hw[0] / 2), math.ceil(image_hw[1] / 2))
    torch.testing.assert_close(actual, expected)
    # More than one tile ensures the assertion covers allocation and placement,
    # rather than only setup-time shape propagation.
    assert len(streaming._last_forward_tiles) > 1


def test_multiple_strided_convs_propagate_full_output_size_layer_by_layer():
    torch.manual_seed(92)
    reference = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(2, 2, kernel_size=3, stride=2, padding=1),
    ).eval()
    streaming_module = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(2, 2, kernel_size=3, stride=2, padding=1),
    ).eval()
    streaming_module.load_state_dict(reference.state_dict())
    streaming = StreamingCNN(
        streaming_module,
        tile_shape=(1, 1, 16, 16),
        deterministic=True,
        copy_to_gpu=False,
    )
    image = torch.randn(1, 1, 33, 35)

    expected = reference(image)
    actual = streaming(image)

    assert actual.shape == expected.shape == (1, 2, 9, 9)
    torch.testing.assert_close(actual, expected)


def test_sshr_saliency_shifted_boundary_tiles_track_writes_for_non_divisible_size():
    """SSHR-style shifted boundary replay tracks writes independently of values."""
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.saliency_map = torch.zeros(1, 3, 11, 13, dtype=torch.double)
    scnn.saliency_coverage_map = torch.zeros_like(scnn.saliency_map, dtype=torch.bool)
    scnn.saliency_nonzero_coverage_map = torch.zeros_like(scnn.saliency_map, dtype=torch.bool)

    input_conv = StreamingConv2d(3, 2, kernel_size=1).double()
    input_conv.grad_lost = Lost(1, 1, 1, 1)
    input_conv.output_stride = torch.tensor([1, 1, 1])
    expected = torch.zeros_like(scnn.saliency_map)

    # This fixed 11x13 fixture is non-divisible by its nominal 6x6 safe step.
    # Consequently the last row starts at 3 (not 6), and the last column at 5
    # (not 6), exercising overlap on both axes just like the SSHR comparison.
    safe_step = (6, 6)
    assert all(size % step for size, step in zip(scnn.saliency_map.shape[-2:], safe_step))
    tile_starts = ((0, 0), (0, 5), (3, 0), (3, 5))
    old_indices = (
        Box(0, 0, 0, 0, None),
        Box(0, 7, 7, 0, None),
        Box(0, 7, 13, 0, None),
        Box(7, 11, 7, 0, None),
    )
    assert tile_starts[-1][0] == 3
    assert tile_starts[-1][1] == 5
    for tile_index, ((input_y, input_x), previously_seen) in enumerate(
        zip(tile_starts, old_indices), start=1
    ):
        sides = Sides(
            left=input_x == 0,
            top=input_y == 0,
            right=input_x == 5,
            bottom=input_y == 3,
        )
        input_conv.input_loc = Box(input_y, 8, input_x, 8, sides)
        scnn.saliency_old_indices = previously_seen
        tile_gradient = torch.full((1, 3, 8, 8), float(tile_index), dtype=torch.double)
        scnn._backward_saliency_hook(
            input_conv,
            (tile_gradient,),
            (torch.ones(1, 2, 8, 8, dtype=torch.double),),
        )
        expected[
            :, :, input_y : input_y + tile_gradient.shape[-2],
            input_x : input_x + tile_gradient.shape[-1],
        ] += tile_gradient

    reference_support = expected.ne(0)
    missing_support = reference_support & ~scnn.saliency_coverage_map
    assert not missing_support.any(), missing_support.nonzero().tolist()
    # Keep numerical disagreement separate from the coverage assertion above.
    assert reference_support.all()
    assert scnn.saliency_coverage_map.all()
    assert scnn.saliency_nonzero_coverage_map.all()
    # Both shifted last-axis tiles contribute in the overlapping corner.
    assert scnn.saliency_map[0, 0, 5, 6].item() == 10
    assert scnn.saliency_map[0, 0, 1, 7].item() == 3
    torch.testing.assert_close(scnn.saliency_map, expected, rtol=0, atol=0)


def test_saliency_coverage_distinguishes_zero_write_from_unvisited_coordinate():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.saliency_map = torch.zeros(1, 3, 4, 5, dtype=torch.double)
    scnn.saliency_coverage_map = torch.zeros_like(scnn.saliency_map, dtype=torch.bool)
    scnn.saliency_nonzero_coverage_map = torch.zeros_like(scnn.saliency_map, dtype=torch.bool)

    input_conv = StreamingConv2d(3, 2, kernel_size=1).double()
    input_conv.grad_lost = Lost(0, 0, 0, 0)
    input_conv.output_stride = torch.tensor([1, 1, 1])
    input_conv.input_loc = Box(0, 4, 0, 4, Sides(True, True, False, True))
    scnn.saliency_old_indices = Box(0, 0, 0, 0, None)
    tile_gradient = torch.zeros(1, 3, 4, 4, dtype=torch.double)
    tile_gradient[..., 1, 2] = 1

    scnn._backward_saliency_hook(
        input_conv,
        (tile_gradient,),
        (torch.ones(1, 2, 4, 4, dtype=torch.double),),
    )

    assert scnn.saliency_coverage_map[..., :4].all()
    assert not scnn.saliency_coverage_map[..., 4].any()
    torch.testing.assert_close(scnn.saliency_nonzero_coverage_map, scnn.saliency_map.ne(0))


def test_saliency_raw_placement_uses_input_coordinates_for_strided_first_conv():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.saliency_map = torch.zeros(1, 3, 12, 13, dtype=torch.double)
    scnn.saliency_coverage_map = torch.zeros_like(scnn.saliency_map, dtype=torch.bool)
    scnn.saliency_nonzero_coverage_map = torch.zeros_like(scnn.saliency_map, dtype=torch.bool)

    input_conv = StreamingConv2d(3, 2, kernel_size=3, stride=2, padding=1).double()
    input_conv.grad_lost = Lost(0, 0, 0, 0)
    input_conv.output_stride = torch.tensor([1, 1, 1])
    input_conv.input_loc = Box(3, 4, 5, 6, Sides(False, False, True, True))
    # The legacy ownership calculation still runs for the diagnostic candidates.
    scnn.saliency_old_indices = Box(1, 0, 2, 0, None)
    raw = torch.arange(72, dtype=torch.double).reshape(1, 3, 4, 6)

    scnn._backward_saliency_hook(
        input_conv,
        (raw,),
        (torch.ones(1, 2, 2, 3, dtype=torch.double),),
    )

    expected = torch.zeros_like(scnn.saliency_map)
    expected[:, :, 3:7, 5:11] = raw
    torch.testing.assert_close(scnn.saliency_map, expected, rtol=0, atol=0)
    assert scnn.saliency_coverage_map[:, :, 3:7, 5:11].all()


def test_saliency_diagnostics_capture_stages_without_changing_production_map():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.saliency_diagnostics = True
    scnn.saliency_map = torch.zeros(1, 3, 6, 6)
    scnn.saliency_coverage_map = torch.zeros_like(scnn.saliency_map, dtype=torch.bool)
    scnn.saliency_nonzero_coverage_map = torch.zeros_like(scnn.saliency_map, dtype=torch.bool)
    scnn.saliency_diagnostic_records = []
    scnn.saliency_diagnostic_maps = {
        name: torch.zeros_like(scnn.saliency_map)
        for name in ("raw", "grad_lost", "ownership", "production")
    }
    scnn.saliency_diagnostic_write_count_maps = {
        name: torch.zeros_like(scnn.saliency_map, dtype=torch.int32)
        for name in ("raw", "grad_lost", "ownership")
    }
    scnn._saliency_diagnostic_destination_coverage = torch.zeros(6, 6, dtype=torch.bool)

    input_conv = StreamingConv2d(3, 2, kernel_size=1)
    input_conv.grad_lost = Lost(1, 1, 1, 1)
    input_conv.output_stride = torch.tensor([1, 1, 1])
    input_conv.input_loc = Box(0, 6, 0, 6, Sides(True, True, False, False))
    scnn.saliency_old_indices = Box(0, 0, 0, 0, None)
    raw = torch.ones(1, 3, 6, 6)

    scnn._backward_saliency_hook(input_conv, (raw,), (torch.ones(1, 2, 6, 6),))

    record = scnn.saliency_diagnostic_records[0]
    assert record["raw_shape"] == (1, 3, 6, 6)
    assert record["post_grad_lost_shape"] == (1, 3, 5, 5)
    assert record["post_ownership_shape"] == (1, 3, 5, 5)
    assert record["raw_nonzero"] == 108
    assert record["destination_overlaps_previous"] is False
    torch.testing.assert_close(scnn.saliency_diagnostic_maps["production"], scnn.saliency_map)
    assert scnn.saliency_diagnostic_write_count_maps["raw"].eq(1).all()
    assert scnn.saliency_diagnostic_write_count_maps["grad_lost"][..., :5, :5].eq(1).all()
    assert scnn.saliency_diagnostic_write_count_maps["grad_lost"][..., 5, :].eq(0).all()
    assert scnn.saliency_diagnostic_write_count_maps["ownership"][..., :5, :5].eq(1).all()


@pytest.mark.cuda_integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_testnet_raw_saliency_candidate_characterizes_later_stage_regressions(tmp_path):
    """Characterize the TestNet result before extending this to ResNet and SSHR."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float64
    rtol, atol = 1e-7, 1e-9

    # These are deliberately the dimensions used by grad_compare_testnet.py,
    # rather than a reduced unit-test proxy: shifted final tiles are part of
    # the diagnostic result being locked down here.
    tile_size = 960
    input_size = 3520
    image = torch.rand((1, 3, input_size, input_size), device=device, dtype=dtype)
    network = SaliencyTestNet(
        tile_size,
        verbose=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
        saliency=True,
        tile_cache_path=tmp_path / "testnet_saliency_tile_cache",
    ).to(device=device, dtype=dtype)
    scnn = network.stream_network
    scnn.device = device
    scnn.dtype = dtype
    scnn.mean = scnn.mean.to(device=device, dtype=dtype)
    scnn.std = scnn.std.to(device=device, dtype=dtype)
    scnn.saliency_diagnostics = True
    for module in scnn.stream_module.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    streamed_output = network(image)
    if not isinstance(streamed_output, torch.Tensor):
        assert len(streamed_output) == 1
        streamed_output = streamed_output[0]
    target = torch.tensor(50.0, device=device, dtype=dtype)
    loss = nn.MSELoss()(torch.sigmoid(streamed_output.mean()), target)
    (output_gradient,) = torch.autograd.grad(loss, streamed_output)
    scnn.backward(image, output_gradient.detach())

    candidates = scnn.saliency_diagnostic_maps
    scnn.disable()
    reference_input = image.detach().clone().requires_grad_(True)
    reference_output = scnn.stream_module(reference_input)
    torch.autograd.backward(reference_output, output_gradient.detach())
    reference = reference_input.grad.detach().cpu()

    raw = candidates["raw"].to(dtype=dtype)
    cropped = candidates["grad_lost"].to(dtype=dtype)
    owned = candidates["ownership"].to(dtype=dtype)
    reference_support = reference.ne(0)

    raw_support = raw.ne(0)
    assert not (reference_support & ~raw_support).any()
    assert not (raw_support & ~reference_support).any()
    torch.testing.assert_close(raw, reference, rtol=rtol, atol=atol)

    # Cropping itself changes values beyond the float64 parity tolerance.
    cropped_error = (cropped - reference).abs()
    cropped_tolerance = atol + rtol * reference.abs()
    assert (cropped_error > cropped_tolerance).any()

    # Ownership selection is a separate, later diagnostic: it removes support
    # that was still present after the side-aware grad_lost crop. Avoid locking
    # down its current count, which can vary with supported PyTorch versions.
    cropped_support = cropped.ne(0)
    owned_support = owned.ne(0)
    additional_missing_support = reference_support & cropped_support & ~owned_support
    assert additional_missing_support.any()


def test_strided_conv_backward_accepts_gap_before_shifted_final_replay_row():
    """A diagnostic cursor gap must not discard any dependency gradients."""
    torch.manual_seed(93)
    streaming = StreamingConv2d(1, 2, kernel_size=3, stride=2, padding=1).double()
    reference = streaming.to_torch_conv2d()
    streaming.output_stride = torch.tensor([1, 1, 1])

    replay_tiles = (
        (torch.randn(1, 1, 9, 7, dtype=torch.double), 0),
        (torch.randn(1, 1, 3, 7, dtype=torch.double), 12),
    )
    expected_input_gradients = []
    actual_input_gradients = []

    for index, (tile, input_y) in enumerate(replay_tiles):
        reference_input = tile.detach().clone().requires_grad_(True)
        streaming_input = tile.detach().clone().requires_grad_(True)
        streaming.input_loc = Box(
            input_y,
            tile.shape[2],
            0,
            tile.shape[3],
            Sides(True, index == 0, True, index == len(replay_tiles) - 1),
        )

        reference_output = reference(reference_input)
        streaming_output = streaming(streaming_input)
        output_gradient = torch.randn_like(reference_output)
        reference_output.backward(output_gradient)
        streaming_output.backward(output_gradient)

        expected_input_gradients.append(reference_input.grad)
        actual_input_gradients.append(streaming_input.grad)
        if index == 0:
            assert streaming.seen_indices.height == 5
        else:
            projected_data_y = input_y // streaming.stride[0]
            assert projected_data_y == 6

    assert streaming.seen_indices.y == 6
    assert streaming.seen_indices.height == 8
    for actual, expected in zip(actual_input_gradients, expected_input_gradients):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(streaming.weight.grad, reference.weight.grad)
    torch.testing.assert_close(streaming.bias.grad, reference.bias.grad)


def test_convert_and_reset_every_supported_layer_type():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn._streaming_reducers = []

    conv = nn.Conv2d(3, 3, kernel_size=1)
    norm = ChannelLayerNorm(3)
    scale = LayerScale((1, 3, 1, 1))
    upsample = nn.Upsample(scale_factor=2, mode="nearest")
    probe = StatisticsProbe()
    merge = StreamingMerge("add")
    model = nn.ModuleList([conv, norm, scale, upsample, probe, merge])

    stats = {
        "grad_lost": Lost(0, 0, 0, 0),
        "output_stride": torch.tensor([1, 1, 1]),
    }
    scnn._module_stats = {
        conv: dict(stats),
        norm: dict(stats),
        scale: dict(stats),
        upsample: dict(stats),
    }

    converted = scnn._convert_modules_for_streaming(model)

    assert isinstance(converted[0], StreamingConv2d)
    assert isinstance(converted[1], StreamingChannelLayerNorm)
    assert isinstance(converted[2], StreamingLayerScale)
    assert isinstance(converted[3], StreamingUpsample2d)
    assert converted[4] is probe
    assert converted[5] is merge

    restored = scnn._reset_converted_modules(converted)

    assert type(restored[0]) is nn.Conv2d
    assert type(restored[1]) is ChannelLayerNorm
    assert type(restored[2]) is LayerScale
    assert type(restored[3]) is nn.Upsample
    assert restored[4] is probe
    assert restored[5] is merge


def test_forward_statistics_store_and_preserve_dilated_conv2d_dilation():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.eps = 1e-5
    scnn.dtype = torch.float32
    scnn.device = torch.device("cpu")
    scnn._saved_tensors = {}
    scnn._module_stats = {}
    scnn._print_verbose = lambda *args, **kwargs: None

    module = nn.Conv2d(3, 4, kernel_size=3, padding=(2, 3), dilation=(2, 3), bias=False)
    scnn.stream_module = nn.Sequential(module)
    inpt = torch.ones(1, 3, 16, 16)

    with torch.no_grad():
        output = module(inpt)
        scnn._forward_gather_statistics_hook(module, (inpt,), output)

    module_stats = scnn._module_stats[module]
    assert module_stats["dilation"] == (1, 2, 3)

    scnn.output_stride = torch.tensor([1, 1, 1])
    scnn.tile_output_lost = Lost(0, 0, 0, 0)
    scnn._tile_output_lost = [scnn.tile_output_lost]
    scnn.tile_gradient_lost = Lost(0, 0, 0, 0)
    scnn._tile_output_shape = torch.Size([1, 4, 16, 16])
    scnn._tile_output_shapes = [scnn._tile_output_shape]
    scnn._output_stride_per_output = [scnn.output_stride]
    scnn._output_spec = ("tensor", None)

    state = scnn.get_tile_cache()
    restored = StreamingCNN.__new__(StreamingCNN)
    restored.stream_module = scnn.stream_module
    restored._module_stats = {}
    restored.disable = lambda: None
    restored.enable = lambda: None

    restored.load_tile_cache(state)

    assert restored._module_stats[module]["dilation"] == (1, 2, 3)

def _conv2d_backward_valid_lost_for_dilation(dilation):
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.eps = 1e-5
    scnn.dtype = torch.float32
    scnn.device = torch.device("cpu")
    scnn._saved_tensors = {}
    scnn._module_stats = {}
    scnn._print_verbose = lambda *args, **kwargs: None

    module = nn.Conv2d(1, 1, kernel_size=2, stride=2, dilation=dilation, bias=False)
    module.weight.data.fill_(1)
    inpt = torch.ones(1, 1, 8, 8, requires_grad=True)
    output = module(inpt)
    scnn._module_stats[module] = {}

    output.sum().backward()
    scnn._backward_gather_statistics_hook(module, (inpt.grad,), (torch.ones_like(output),))

    return scnn._module_stats[module]["backward_valid_lost"]


def test_backward_statistics_use_dilated_effective_kernel_for_overlap():
    dilation_1_lost = _conv2d_backward_valid_lost_for_dilation(1)
    dilation_2_lost = _conv2d_backward_valid_lost_for_dilation(2)

    assert dilation_1_lost == Lost(0, 0, 0, 0)
    assert dilation_2_lost.top > dilation_1_lost.top
    assert dilation_2_lost.left > dilation_1_lost.left
    assert dilation_2_lost.bottom > dilation_1_lost.bottom
    assert dilation_2_lost.right > dilation_1_lost.right


def test_batch_norm_checkpoint_state_is_restored_and_does_not_affect_tile_cache():
    def configure(running_mean, running_var):
        model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
        ).eval()
        with torch.no_grad():
            model[1].weight.copy_(torch.tensor([0.5, 1.5, 2.5, 3.5]))
            model[1].bias.copy_(torch.tensor([-1.0, -0.5, 0.5, 1.0]))
            model[1].running_mean.copy_(torch.tensor(running_mean))
            model[1].running_var.copy_(torch.tensor(running_var))
            model[1].num_batches_tracked.fill_(17)
        checkpoint_state = {name: value.clone() for name, value in model.state_dict().items()}

        scnn = StreamingCNN(
            model,
            tile_shape=(1, 3, 8, 8),
            deterministic=True,
            copy_to_gpu=False,
            statistics_on_cpu=False,
            normalize_on_gpu=False,
        )

        restored_state = scnn.stream_module.state_dict()
        assert restored_state.keys() == checkpoint_state.keys()
        for name, expected in checkpoint_state.items():
            torch.testing.assert_close(restored_state[name], expected, rtol=0, atol=0)
        return scnn.get_tile_cache()

    first_cache = configure(
        running_mean=[-100.0, -2.0, 5.0, 80.0],
        running_var=[0.01, 0.5, 10.0, 250.0],
    )
    second_cache = configure(
        running_mean=[90.0, 7.0, -4.0, -120.0],
        running_var=[300.0, 20.0, 0.25, 0.02],
    )

    def assert_identical_finite(first, second):
        assert type(first) is type(second)
        if isinstance(first, torch.Tensor):
            assert torch.isfinite(first).all()
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            assert first.keys() == second.keys()
            for key in first:
                assert_identical_finite(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            assert len(first) == len(second)
            for first_value, second_value in zip(first, second):
                assert_identical_finite(first_value, second_value)
        elif isinstance(first, nn.Module):
            assert repr(first) == repr(second)
        elif isinstance(first, float):
            assert math.isfinite(first)
            assert first == second
        else:
            assert first == second

    assert_identical_finite(first_cache, second_cache)


@pytest.mark.parametrize(
    "reducer_cls",
    [
        AttentionGeMReducer,
        StreamingAttentionGeMReducer,
        FusedAttentionGeMReducer,
        StreamingFusedAttentionGeMReducer,
    ],
)
@pytest.mark.parametrize("bad_eps", [-0.01, 1.01, float("inf"), float("nan")])
def test_attention_gem_uniform_attention_eps_validation(reducer_cls, bad_eps):
    with pytest.raises(ValueError, match="uniform_attention_eps"):
        reducer_cls(uniform_attention_eps=bad_eps)


def test_attention_gem_uniform_attention_eps_conversion_round_trip_preserves_value():
    reducer = AttentionGeMReducer(r_init=2.25, eps=1e-5, uniform_attention_eps=0.125)

    streaming = reducer.to_streaming()
    round_tripped = streaming.to_reducer()

    assert isinstance(streaming, StreamingAttentionGeMReducer)
    assert streaming.uniform_attention_eps == pytest.approx(0.125)
    assert round_tripped.uniform_attention_eps == pytest.approx(0.125)
    assert reducer.uniform_attention_eps == pytest.approx(0.125)


def test_fused_attention_gem_uniform_attention_eps_conversion_round_trip_preserves_value():
    reducer = FusedAttentionGeMReducer(
        r_init=2.25,
        eps=1e-5,
        value_weights=(0.2, 0.5, 0.3),
        attention_weights=(0.1, 0.7, 0.2),
        uniform_attention_eps=0.125,
    )

    streaming = reducer.to_streaming()
    round_tripped = streaming.to_reducer()

    assert isinstance(streaming, StreamingFusedAttentionGeMReducer)
    assert streaming.uniform_attention_eps == pytest.approx(0.125)
    assert round_tripped.uniform_attention_eps == pytest.approx(0.125)
    assert reducer.uniform_attention_eps == pytest.approx(0.125)


def test_attention_gem_uniform_attention_eps_mixes_attention_and_uniform_valid_means():
    x = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
    logits = torch.tensor([[[[0.0, 2.0], [-3.0, 10.0]]]])
    mask = torch.tensor([[True, False], [True, True]])
    reducer = AttentionGeMReducer(r_init=1.0, eps=1e-6, uniform_attention_eps=0.25)

    actual = reducer(x, logits, mask=mask)

    valid_x = torch.tensor([1.0, 4.0, 8.0])
    valid_logits = torch.tensor([0.0, -3.0, 10.0])
    attention_mean = (torch.softmax(valid_logits, dim=0) * valid_x).sum().view(1, 1, 1, 1)
    uniform_mean = valid_x.mean().view(1, 1, 1, 1)
    expected = 0.75 * attention_mean + 0.25 * uniform_mean
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_attention_gem_uniform_attention_eps_default_matches_explicit_zero():
    torch.manual_seed(133)
    x = torch.rand(2, 3, 4, 5) + 0.05
    logits = torch.randn(2, 1, 4, 5)
    y1 = torch.rand(2, 3, 4, 5) + 0.05
    y2 = torch.rand(2, 3, 4, 5) + 0.05
    y3 = torch.rand(2, 3, 4, 5) + 0.05
    logits_triplet = [torch.randn(2, 1, 4, 5) for _ in range(3)]

    default_attention = AttentionGeMReducer(r_init=2.0, eps=1e-6)
    zero_attention = AttentionGeMReducer(r_init=2.0, eps=1e-6, uniform_attention_eps=0.0)
    default_fused = FusedAttentionGeMReducer(r_init=2.0, eps=1e-6)
    zero_fused = FusedAttentionGeMReducer(r_init=2.0, eps=1e-6, uniform_attention_eps=0.0)

    assert default_attention.uniform_attention_eps == 0.0
    assert default_fused.uniform_attention_eps == 0.0
    assert torch.allclose(default_attention(x, logits), zero_attention(x, logits), atol=0, rtol=0)
    assert torch.allclose(
        default_fused(y1, y2, y3, *logits_triplet),
        zero_fused(y1, y2, y3, *logits_triplet),
        atol=0,
        rtol=0,
    )


def test_fused_attention_gem_uniform_attention_eps_adds_uniform_term_after_branch_fusion():
    y1 = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
    y2 = torch.tensor([[[[3.0, 5.0], [7.0, 11.0]]]])
    y3 = torch.tensor([[[[13.0, 17.0], [19.0, 23.0]]]])
    logits = (
        torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]),
        torch.tensor([[[[3.0, 2.0], [1.0, 0.0]]]]),
        torch.tensor([[[[0.5, -0.5], [1.5, -1.5]]]]),
    )
    value_weights = (0.5, 0.25, 0.25)
    attention_weights = (0.2, 0.3, 1.0)
    reducer = FusedAttentionGeMReducer(
        r_init=1.0,
        eps=1e-6,
        value_weights=value_weights,
        attention_weights=attention_weights,
        uniform_attention_eps=0.4,
    )

    actual = reducer(y1, y2, y3, *logits)

    fused_y = value_weights[0] * y1 + value_weights[1] * y2 + value_weights[2] * y3
    branch_means = [
        (torch.softmax(logit.flatten(), dim=0) * fused_y.flatten()).sum().view(1, 1, 1, 1)
        for logit in logits
    ]
    attention_mean = sum(weight * branch for weight, branch in zip(attention_weights, branch_means))
    uniform_mean = fused_y.mean(dim=(-2, -1), keepdim=True)
    expected = 0.6 * attention_mean + 0.4 * uniform_mean
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


class AllReducerHeadsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.sum_head = nn.Sequential(nn.Conv2d(6, 2, kernel_size=1, bias=False), SumReducer())
        self.mean_head = nn.Sequential(nn.Conv2d(6, 3, kernel_size=1, bias=False), MeanReducer())

    def forward(self, x):
        feat = self.backbone(x)
        return self.sum_head(feat), self.mean_head(feat)


class DownsampledReducerNet(nn.Module):
    def __init__(self, mask_resize: bool = False):
        super().__init__()
        self.down = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.reducer = MeanReducer(mask_resize=mask_resize)

    def forward(self, x, mask: torch.Tensor | None = None):
        feat = self.down(x)
        return self.reducer(feat, mask=mask)


class MultiResolutionReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.half = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.quarter = nn.Conv2d(3, 5, kernel_size=3, stride=4, padding=1, bias=False)
        self.half_reducer = MeanReducer(mask_resize=True)
        self.quarter_reducer = MeanReducer(mask_resize=True)

    def forward(self, x):
        return self.half_reducer(self.half(x)), self.quarter_reducer(self.quarter(x))


class MixedHeadsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.raw_head = nn.Conv2d(5, 4, kernel_size=1, bias=False)
        self.reducer_head = nn.Sequential(nn.Conv2d(5, 4, kernel_size=1, bias=False), MeanReducer())

    def forward(self, x):
        feat = self.backbone(x)
        return {"raw": self.raw_head(feat), "reduced": self.reducer_head(feat)}


class GeMHeadNet(nn.Module):
    def __init__(self, learnable_r: bool):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.gem_head = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=1, bias=False),
            GeMReducer(r=3.2, eps=1e-6, learnable_r=learnable_r),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.gem_head(feat)


class SharedRGeMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(4.0))
        self.red1 = GeMReducer(r_parameter=self.r)
        self.red2 = GeMReducer(r_parameter=self.r)
        self.red3 = GeMReducer(r_parameter=self.r)
        self.red4 = GeMReducer(r_parameter=self.r)

    def forward(self, x, mask=None):
        x = x.clamp_min(1e-3)
        return (
            self.red1(x, mask=mask),
            self.red2(x + 0.1, mask=mask),
            self.red3(x + 0.2, mask=mask),
            self.red4(x + 0.3, mask=mask),
        )


class AttentionGeMHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(5, 2, kernel_size=1, bias=True),
            nn.Softplus(),
        )
        self.att_logits = nn.Conv2d(5, 1, kernel_size=1, bias=True)
        self.reducer = AttentionGeMReducer(r_init=2.0, eps=1e-6)

    def forward(self, x, mask: torch.Tensor | None = None):
        feat = self.backbone(x)
        value = self.value_head(feat)
        logits = self.att_logits(feat)
        reduced = self.reducer(value, logits, mask=mask)
        return reduced, value, logits


class NGWPMultiHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
        )
        self.score_heads = nn.ModuleList([nn.Conv2d(5, 2, kernel_size=1) for _ in range(4)])
        self.activation_heads = nn.ModuleList([nn.Conv2d(5, 2, kernel_size=1) for _ in range(4)])
        self.reducers = nn.ModuleList([NGWPReducer(eps=1e-6) for _ in range(4)])

    def forward(self, x, mask: torch.Tensor | None = None):
        feat = self.backbone(x)
        scores = [head(feat) for head in self.score_heads]
        activation_masks = [torch.sigmoid(head(feat)) for head in self.activation_heads]
        return tuple(
            reducer(score, activation_mask, mask=mask)
            for reducer, score, activation_mask in zip(self.reducers, scores, activation_masks)
        )



class FusedAttentionGeMHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
        )
        self.value_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(5, 2, kernel_size=1, bias=True), nn.Softplus()),
                nn.Sequential(nn.Conv2d(5, 2, kernel_size=1, bias=True), nn.Softplus()),
                nn.Sequential(nn.Conv2d(5, 2, kernel_size=1, bias=True), nn.Softplus()),
            ]
        )
        self.logit_heads = nn.ModuleList([nn.Conv2d(5, 1, kernel_size=1, bias=True) for _ in range(3)])
        self.reducer = FusedAttentionGeMReducer(r_init=2.5, eps=1e-6)

    def forward(self, x, mask: torch.Tensor | None = None):
        feat = self.backbone(x)
        values = [head(feat) for head in self.value_heads]
        logits = [head(feat) for head in self.logit_heads]
        return self.reducer(*values, *logits, mask=mask)


class ValueLogitsReducer(BaseReducer):
    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError(f"ValueLogitsReducer expects exactly two inputs, got {len(inputs)}")
        value, logits = inputs
        if value.ndim != 4 or logits.ndim != 4:
            raise ValueError("ValueLogitsReducer expects NCHW inputs")
        if value.shape != logits.shape:
            raise ValueError(f"ValueLogitsReducer shape mismatch value={tuple(value.shape)} logits={tuple(logits.shape)}")
        if self._streaming_passthrough:
            self._last_inputs = inputs
            self._last_output = value
            return value
        weights = torch.softmax(logits, dim=-1)
        if mask is not None:
            mask = mask.to(device=value.device, dtype=value.dtype)[None, None]
            weights = weights * mask
        return (value * weights).sum(dim=(-2, -1), keepdim=True)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        return StreamingValueLogitsReducer()


class StreamingValueLogitsReducer(BaseStreamingGlobalReducer):
    def __init__(self):
        super().__init__(mode="sum")

    def init_reduction_state(self, *, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, accumulator_dtype: torch.dtype) -> None:
        self.running_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=dtype)

    def accumulate_valid_tile(self, tile: torch.Tensor | tuple[torch.Tensor, ...], valid_mask: torch.Tensor) -> None:
        value, logits = self._parse_multi_input_payload(tile)
        weights = torch.softmax(logits, dim=-1)
        mask4 = valid_mask.to(device=value.device, dtype=value.dtype)[None, None]
        self.running_sum = self.running_sum + (value * weights * mask4).sum(dim=(-2, -1), keepdim=True)

    def finalize_from_state(self) -> torch.Tensor:
        return self.running_sum

    def reduce_tile_for_backward(self, trimmed_output: torch.Tensor | tuple[torch.Tensor, ...], valid_mask: torch.Tensor | None, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        value, logits = self._parse_multi_input_payload(trimmed_output)
        weights = torch.softmax(logits, dim=-1)
        if valid_mask is not None:
            mask4 = valid_mask.to(device=value.device, dtype=value.dtype)[None, None]
            return (value * weights * mask4).sum(dim=(-2, -1), keepdim=True)
        return (value * weights).sum(dim=(-2, -1), keepdim=True)


class ValueLogitsHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False), nn.ReLU())
        self.value_head = nn.Conv2d(6, 3, kernel_size=1, bias=False)
        self.logit_head = nn.Conv2d(6, 3, kernel_size=1, bias=False)
        self.reducer = ValueLogitsReducer()

    def forward(self, x, mask: torch.Tensor | None = None):
        feat = self.backbone(x)
        value = self.value_head(feat)
        logits = self.logit_head(feat)
        reduced = self.reducer(value, logits, mask=mask)
        return reduced, value, logits


def _make_streaming(model: nn.Module, tile_size: int = 4):
    constructor = StreamingConstructor(
        model,
        tile_size=tile_size,
        verbose=False,
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=True,
        normalize_on_gpu=False,
    )
    return constructor.prepare_streaming_model()


def _seen_indices_by_module(scnn: StreamingCNN):
    return {
        name: (
            module.seen_indices.y,
            module.seen_indices.height,
            module.seen_indices.x,
            module.seen_indices.width,
            module.seen_indices.sides,
        )
        for name, module in scnn.stream_module.named_modules()
        if hasattr(module, "seen_indices")
    }


def test_scnn_repeated_forward_backward_cycles_and_cached_statistics_reset_state():
    torch.manual_seed(5)

    def make_model():
        return nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=1, bias=True),
            nn.Softplus(),
            nn.Conv2d(5, 2, kernel_size=1, bias=True),
        ).eval()

    source = make_model()
    source_state = {name: value.detach().clone() for name, value in source.state_dict().items()}

    scnn = StreamingCNN(
        source,
        tile_shape=(1, 3, 4, 4),
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
    )
    scnn.gather_input_gradient = True
    scnn._remove_hooks()
    scnn._add_hooks_for_streaming()

    cached_source = make_model()
    cached_source.load_state_dict(source_state)
    cached_scnn = StreamingCNN(
        cached_source,
        tile_shape=(1, 3, 4, 4),
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
        state_dict=scnn.get_tile_cache(),
    )
    cached_scnn.gather_input_gradient = True
    cached_scnn._remove_hooks()
    cached_scnn._add_hooks_for_streaming()

    def run_cycle(streaming_model: StreamingCNN, cycle: int):
        streaming_model.zero_grad(set_to_none=True)
        stream_input = (torch.rand(1, 3, 9, 11) + 0.05).requires_grad_(True)

        reference = make_model()
        reference.load_state_dict(source_state)
        reference.zero_grad(set_to_none=True)
        reference_input = stream_input.detach().clone().requires_grad_(True)
        reference_output = reference(reference_input)
        output_gradient = torch.full_like(reference_output, 0.19 + cycle * 0.11)
        torch.autograd.backward(reference_output, output_gradient)

        streaming_output = streaming_model.forward(stream_input)
        streaming_model.backward(stream_input, output_gradient.detach().clone())

        torch.testing.assert_close(streaming_output, reference_output.detach(), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(streaming_model.saliency_map, reference_input.grad, rtol=1e-4, atol=1e-5)

        streaming_parameters = dict(streaming_model.stream_module.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        assert streaming_parameters.keys() == reference_parameters.keys()
        for name, reference_parameter in reference_parameters.items():
            assert reference_parameter.grad is not None, name
            assert streaming_parameters[name].grad is not None, name
            torch.testing.assert_close(
                streaming_parameters[name].grad,
                reference_parameter.grad,
                rtol=1e-4,
                atol=1e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )

        return _seen_indices_by_module(streaming_model)

    first_reset_state = run_cycle(scnn, cycle=0)
    second_reset_state = run_cycle(scnn, cycle=1)
    cached_reset_state = run_cycle(cached_scnn, cycle=2)

    assert first_reset_state
    assert second_reset_state == first_reset_state
    assert cached_reset_state == first_reset_state


def test_scnn_forward_all_reducer_heads_parity():
    torch.manual_seed(7)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 9, 11)

    with torch.no_grad():
        expected_sum, expected_mean = model(image)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image)

    assert streamed_sum.shape == expected_sum.shape
    assert streamed_mean.shape == expected_mean.shape
    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_scnn_forward_mixed_reducer_non_reducer_heads_parity():
    torch.manual_seed(11)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 10, 12)

    with torch.no_grad():
        expected = model(image)

    scnn = _make_streaming(model, tile_size=5)
    with torch.no_grad():
        streamed = scnn.forward(image)

    assert streamed["raw"].shape == expected["raw"].shape
    assert streamed["reduced"].shape == expected["reduced"].shape
    assert torch.allclose(streamed["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)


def test_scnn_forward_border_tiles_parity_small_image_large_tile():
    torch.manual_seed(3)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 3, 4)

    with torch.no_grad():
        expected = model(image)

    scnn = _make_streaming(model, tile_size=8)
    with torch.no_grad():
        streamed = scnn.forward(image)

    assert torch.allclose(streamed["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)


def test_scnn_backward_reducer_replay_consumed_debug_mode():
    torch.manual_seed(23)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 9, 11)

    scnn = _make_streaming(model, tile_size=4)
    scnn.debug_reducer_replay = True

    with torch.no_grad():
        streamed_output = scnn.forward(image)

    grad = tuple(torch.ones_like(head) for head in streamed_output)
    # no exception means replay assignments were consumed and validated
    scnn.backward(image, grad)


def test_scnn_backward_structure_mismatch_error():
    torch.manual_seed(29)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 6, 7)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed = scnn.forward(image)

    wrong_grad = (torch.ones_like(streamed["raw"]), torch.ones_like(streamed["reduced"]))
    with pytest.raises(ValueError, match="Gradient output structure does not match"):
        scnn.backward(image, wrong_grad)


def test_scnn_backward_reducer_head_gradient_shape_mismatch_error():
    torch.manual_seed(31)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 8, 9)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed = scnn.forward(image)

    wrong_grad = {
        "raw": torch.ones_like(streamed["raw"]),
        "reduced": torch.ones((streamed["reduced"].shape[0], streamed["reduced"].shape[1], 2, 1)),
    }
    with pytest.raises(ValueError, match="Reducer-backed head expects gradient"):
        scnn.backward(image, wrong_grad)


def test_scnn_backward_requires_forward_for_reducer_heads():
    torch.manual_seed(41)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 9, 11)

    scnn = _make_streaming(model, tile_size=4)

    with torch.no_grad():
        expected_output = model(image)

    grad = tuple(torch.ones_like(head) for head in expected_output)
    with pytest.raises(RuntimeError, match="requires prior streaming forward pass"):
        scnn.backward(image, grad)


@pytest.mark.parametrize(
    ("head_module_name", "expected_streaming_type"),
    [
        ("sum_head.1", StreamingSumReducer),
        ("mean_head.1", StreamingMeanReducer),
    ],
)
def test_scnn_converts_reducer_heads_to_streaming_types(head_module_name, expected_streaming_type):
    model = AllReducerHeadsNet().eval()
    scnn = _make_streaming(model, tile_size=4)
    assert isinstance(scnn.stream_module.get_submodule(head_module_name), expected_streaming_type)


@pytest.mark.parametrize("learnable_r", [False, True])
def test_scnn_gem_conversion_forward_backward_parity(learnable_r):
    torch.manual_seed(53 if learnable_r else 47)
    model = GeMHeadNet(learnable_r=learnable_r).eval()
    reference = GeMHeadNet(learnable_r=learnable_r).eval()
    reference.load_state_dict(model.state_dict())

    image = (torch.rand(1, 3, 9, 11) + 0.05).requires_grad_(True)
    ref_image = image.detach().clone().requires_grad_(True)

    ref_out = reference(ref_image)
    ref_grad = torch.full_like(ref_out, 0.37)
    torch.autograd.backward(ref_out, ref_grad)

    scnn = _make_streaming(model, tile_size=4)
    assert isinstance(scnn.stream_module.get_submodule("gem_head.1"), StreamingGeMReducer)
    stream_out = scnn.forward(image.detach().clone())
    assert torch.allclose(stream_out, ref_out.detach(), atol=1e-5, rtol=1e-4)

    scnn.backward(image.detach().clone(), ref_grad.detach().clone())

    stream_parameters = dict(scnn.stream_module.named_parameters())
    stream_grads = {name: p.grad for name, p in stream_parameters.items() if p.grad is not None}
    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    for name, ref_param_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_param_grad, atol=1e-5, rtol=1e-4), name

    # input-grad parity under real SCNN execution
    input_scnn = _make_streaming(GeMHeadNet(learnable_r=learnable_r).eval(), tile_size=4)
    input_scnn.gather_input_gradient = True
    input_scnn._remove_hooks()
    input_scnn._add_hooks_for_streaming()
    scnn_image = image.detach().clone()
    _ = input_scnn.forward(scnn_image)
    input_scnn.backward(scnn_image, ref_grad.detach().clone())
    assert torch.allclose(input_scnn.saliency_map, ref_image.grad, atol=1e-5, rtol=1e-4)

    if learnable_r:
        assert reference.gem_head[1].r.grad is not None
        assert "gem_head.1.r" in stream_parameters
        assert stream_grads["gem_head.1.r"] is not None
        assert torch.allclose(stream_grads["gem_head.1.r"], reference.gem_head[1].r.grad, atol=1e-5, rtol=1e-4)


def test_scnn_shared_r_gem_reducers_forward_backward_parity():
    torch.manual_seed(59)
    model = SharedRGeMNet().eval()
    reference = SharedRGeMNet().eval()
    reference.load_state_dict(model.state_dict())

    assert model.red1.r is model.r
    assert model.red2.r is model.r
    assert model.red3.r is model.r
    assert model.red4.r is model.r

    image = (torch.rand(1, 3, 9, 11) + 0.05).requires_grad_(True)
    ref_image = image.detach().clone().requires_grad_(True)
    mask = torch.rand(9, 11) > 0.2

    ref_out = reference(ref_image, mask=mask)
    ref_grad = tuple(torch.full_like(out, 0.13 + 0.07 * idx) for idx, out in enumerate(ref_out))
    torch.autograd.backward(ref_out, ref_grad)

    scnn = _make_streaming(model, tile_size=4)
    stream = scnn.stream_module

    assert isinstance(stream.red1, StreamingGeMReducer)
    assert isinstance(stream.red2, StreamingGeMReducer)
    assert isinstance(stream.red3, StreamingGeMReducer)
    assert isinstance(stream.red4, StreamingGeMReducer)
    assert stream.red1.r is stream.red2.r
    assert stream.red1.r is stream.red3.r
    assert stream.red1.r is stream.red4.r

    r_param_ids = {
        id(module.r)
        for module in [stream.red1, stream.red2, stream.red3, stream.red4]
    }
    assert len(r_param_ids) == 1

    stream_out = scnn.forward(image.detach().clone(), mask=mask)
    for actual, expected in zip(stream_out, ref_out):
        assert torch.allclose(actual, expected.detach(), atol=1e-5, rtol=1e-4)

    scnn.backward(image.detach().clone(), tuple(grad.detach().clone() for grad in ref_grad), mask=mask)
    assert stream.red1.r.grad is not None
    assert reference.r.grad is not None
    assert torch.allclose(stream.red1.r.grad, reference.r.grad, atol=1e-5, rtol=1e-4)

    input_scnn = _make_streaming(SharedRGeMNet().eval(), tile_size=4)
    input_scnn.stream_module.load_state_dict(reference.state_dict())
    input_scnn.gather_input_gradient = True
    input_scnn._remove_hooks()
    input_scnn._add_hooks_for_streaming()
    scnn_image = image.detach().clone()
    input_stream_out = input_scnn.forward(scnn_image, mask=mask)
    for actual, expected in zip(input_stream_out, ref_out):
        assert torch.allclose(actual, expected.detach(), atol=1e-5, rtol=1e-4)
    input_scnn.backward(scnn_image, tuple(grad.detach().clone() for grad in ref_grad), mask=mask)
    assert torch.allclose(input_scnn.saliency_map, ref_image.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(input_scnn.stream_module.red1.r.grad, reference.r.grad, atol=1e-5, rtol=1e-4)


def test_gem_reducer_learnable_r_constructor_registers_parameter():
    reducer = GeMReducer(r_init=2.75, learnable_r=True)

    assert isinstance(reducer.r, torch.nn.Parameter)
    assert reducer.learnable_r is True
    assert torch.allclose(reducer.r.detach(), torch.tensor(2.75))
    assert dict(reducer.named_parameters())["r"] is reducer.r


def test_gem_reducer_legacy_r_constructor_registers_learnable_parameter():
    reducer = GeMReducer(r=3.25, r_init=1.5, learnable_r=True)

    assert isinstance(reducer.r, torch.nn.Parameter)
    assert reducer.learnable_r is True
    assert torch.allclose(reducer.r.detach(), torch.tensor(3.25))
    assert dict(reducer.named_parameters())["r"] is reducer.r


def test_gem_reducer_accepts_shared_r_parameter():
    shared_r = torch.nn.Parameter(torch.tensor(3.5))

    first = GeMReducer(r_parameter=shared_r)
    second = GeMReducer(r_parameter=shared_r, r_init=1.0, learnable_r=False, r=2.0)

    assert first.r is shared_r
    assert second.r is shared_r
    assert first.learnable_r is True
    assert second.learnable_r is True
    assert dict(first.named_parameters())["r"] is shared_r
    assert dict(second.named_parameters())["r"] is shared_r


def test_streaming_gem_reducer_accepts_shared_r_parameter_with_separate_state():
    shared_r = torch.nn.Parameter(torch.tensor(3.5))

    first = StreamingGeMReducer(r_parameter=shared_r)
    second = StreamingGeMReducer(r_parameter=shared_r, r_init=1.0, learnable_r=False, r=2.0)

    assert first.r is shared_r
    assert second.r is shared_r
    assert first.learnable_r is True
    assert second.learnable_r is True
    assert dict(first.named_parameters())["r"] is shared_r
    assert dict(second.named_parameters())["r"] is shared_r

    first.reset_stream_state(batch_size=1, channels=2, device=torch.device("cpu"), dtype=torch.float32)
    second.reset_stream_state(batch_size=1, channels=2, device=torch.device("cpu"), dtype=torch.float32)

    assert first.running_sum is not second.running_sum
    assert first.running_q is not second.running_q
    assert first.running_count is not second.running_count
    assert first.running_sum.data_ptr() != second.running_sum.data_ptr()
    assert first.running_q.data_ptr() != second.running_q.data_ptr()
    assert first.running_count.data_ptr() != second.running_count.data_ptr()


def test_gem_reducer_to_streaming_shares_learnable_r_only():
    reducer = GeMReducer(r_init=2.75, learnable_r=True)

    streaming_reducer = reducer.to_streaming()

    assert isinstance(streaming_reducer, StreamingGeMReducer)
    assert streaming_reducer.r is reducer.r
    assert streaming_reducer.learnable_r is True
    assert dict(streaming_reducer.named_parameters())["r"] is reducer.r

    streaming_reducer.reset_stream_state(batch_size=1, channels=2, device=torch.device("cpu"), dtype=torch.float32)
    assert streaming_reducer.running_sum is not reducer.r
    assert streaming_reducer.running_q is not reducer.r
    assert streaming_reducer.running_count is not reducer.r


def test_scnn_mixed_head_reducer_mapping_stable():
    torch.manual_seed(61)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 13, 9)

    scnn = _make_streaming(model, tile_size=4)
    assert isinstance(scnn.stream_module.reducer_head[1], StreamingMeanReducer)
    with torch.no_grad():
        expected = model(image)
        stream_first = scnn.forward(image)
        stream_second = scnn.forward(image)

    assert scnn._reducer_head_map
    reducer_head_index = next(iter(scnn._reducer_head_map.keys()))
    assert reducer_head_index == 1
    assert torch.allclose(stream_first["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(stream_first["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(stream_second["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(stream_second["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)


def test_scnn_multi_input_reducer_forward_odd_borders_with_mask_parity():
    torch.manual_seed(101)
    model = ValueLogitsHeadNet().eval()
    image = torch.randn(1, 3, 9, 13)
    mask = (torch.arange(9)[:, None] + torch.arange(13)[None, :]) % 3 != 0

    with torch.no_grad():
        expected, _, _ = model(image, mask=mask)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed, _, _ = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed, expected, atol=1e-5, rtol=1e-4)


def test_scnn_multi_input_reducer_backward_positional_input_parity_streaming_vs_nonstreaming():
    torch.manual_seed(103)
    model = ValueLogitsHeadNet().eval()
    reference = ValueLogitsHeadNet().eval()
    reference.load_state_dict(model.state_dict())
    image = torch.randn(1, 3, 11, 9)
    mask = (torch.rand(11, 9) > 0.25)

    ref_image = image.clone().requires_grad_(True)
    ref_reduced, ref_value, ref_logits = reference(ref_image, mask=mask)
    grad_reduced = torch.full_like(ref_reduced, 0.21)
    grad_value = torch.zeros_like(ref_value)
    grad_logits = torch.zeros_like(ref_logits)
    torch.autograd.backward((ref_reduced, ref_value, ref_logits), (grad_reduced, grad_value, grad_logits))
    ref_grads = {name: p.grad.detach().clone() for name, p in reference.named_parameters() if p.grad is not None}

    scnn = _make_streaming(model, tile_size=5)
    stream_reduced, stream_value, stream_logits = scnn.forward(image.clone(), mask=mask)
    assert torch.allclose(stream_reduced, ref_reduced.detach(), atol=1e-5, rtol=1e-4)
    scnn.backward(image.clone(), (grad_reduced, grad_value, grad_logits), mask=mask)

    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}
    for name, ref_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_grad, atol=1e-5, rtol=1e-4), name


def test_attention_gem_streaming_passthrough_logit_bias_gradient_uniform_shift_parity():
    torch.manual_seed(123)
    reference = AttentionGeMHeadNet().eval()
    streaming_probe = AttentionGeMHeadNet().eval()
    streaming_probe.load_state_dict(reference.state_dict())
    image = torch.randn(1, 3, 11, 13)
    mask = torch.rand(11, 13) > 0.2

    ref_reduced, ref_value, ref_logits = reference(image.clone().requires_grad_(True), mask=mask)
    torch.autograd.backward(
        (ref_reduced, ref_value, ref_logits),
        (torch.ones_like(ref_reduced), torch.zeros_like(ref_value), torch.zeros_like(ref_logits)),
    )
    ref_bias_grad = reference.att_logits.bias.grad.detach().clone()

    streaming_probe.reducer._streaming_passthrough = True
    (stream_value, stream_logits), public_value, public_logits = streaming_probe(image.clone().requires_grad_(True), mask=mask)
    torch.autograd.backward(
        (stream_value, stream_logits, public_value, public_logits),
        (
            torch.ones_like(stream_value),
            torch.zeros_like(stream_logits),
            torch.zeros_like(public_value),
            torch.zeros_like(public_logits),
        ),
    )
    stream_bias_grad = streaming_probe.att_logits.bias.grad.detach().clone()

    assert streaming_probe.reducer._last_inputs[0] is stream_value
    assert streaming_probe.reducer._last_inputs[1] is stream_logits
    assert streaming_probe.reducer._last_output is stream_value
    assert torch.allclose(stream_bias_grad, ref_bias_grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(ref_bias_grad, torch.zeros_like(ref_bias_grad), atol=1e-6, rtol=0)
    assert torch.allclose(stream_bias_grad, torch.zeros_like(stream_bias_grad), atol=1e-6, rtol=0)


def test_ngwp_passthrough_preserves_two_input_payload_order():
    scores = torch.randn(1, 2, 4, 5)
    activation_masks = torch.rand(1, 2, 4, 5)
    reducer = NGWPReducer()
    reducer._streaming_passthrough = True

    offline_payload = reducer(scores, activation_masks)
    streaming = reducer.to_streaming()
    streaming_payload = streaming(scores, activation_masks)

    for owner, payload in ((reducer, offline_payload), (streaming, streaming_payload)):
        assert len(payload) == 2
        assert torch.equal(payload[0], scores)
        assert torch.equal(payload[1], activation_masks)
        assert payload[0].data_ptr() == scores.data_ptr()
        assert payload[1].data_ptr() == activation_masks.data_ptr()
        assert owner._last_inputs is payload
        assert owner._last_output is payload[0]


def test_scnn_four_ngwp_heads_regenerate_eight_tensor_tile_payload_and_forward_parity():
    torch.manual_seed(137)
    model = NGWPMultiHeadNet().eval()
    image = torch.randn(1, 3, 9, 11)
    mask = torch.rand(9, 11) > 0.2

    with torch.no_grad():
        expected = model(image, mask=mask)

    scnn = _make_streaming(model, tile_size=4)

    assert all(isinstance(reducer, StreamingNGWPReducer) for reducer in scnn.stream_module.reducers)
    assert len(scnn._tile_output_shapes) == 8

    with torch.no_grad():
        streamed = scnn.forward(image, mask=mask)

    assert len(streamed) == 4
    for actual, wanted in zip(streamed, expected):
        assert torch.allclose(actual, wanted, atol=1e-5, rtol=1e-4)


def test_scnn_four_ngwp_heads_backward_replay_gradient_parity_for_both_inputs():
    torch.manual_seed(139)
    model = NGWPMultiHeadNet().eval()
    reference = NGWPMultiHeadNet().eval()
    reference.load_state_dict(model.state_dict())
    image = torch.randn(1, 3, 11, 9)
    mask = torch.rand(11, 9) > 0.25

    reference_outputs = reference(image.clone().requires_grad_(True), mask=mask)
    output_gradients = tuple(
        torch.full_like(output, 0.11 + 0.07 * index)
        for index, output in enumerate(reference_outputs)
    )
    torch.autograd.backward(reference_outputs, output_gradients)

    scnn = _make_streaming(model, tile_size=5)
    streamed_outputs = scnn.forward(image.clone(), mask=mask)
    for actual, wanted in zip(streamed_outputs, reference_outputs):
        assert torch.allclose(actual, wanted.detach(), atol=1e-5, rtol=1e-4)
    scnn.backward(image.clone(), output_gradients, mask=mask)

    reference_parameters = dict(reference.named_parameters())
    streaming_parameters = dict(scnn.stream_module.named_parameters())
    for family in ("score_heads", "activation_heads"):
        family_names = [name for name in reference_parameters if name.startswith(family)]
        assert family_names
        for name in family_names:
            assert reference_parameters[name].grad is not None
            assert streaming_parameters[name].grad is not None
            assert torch.allclose(
                streaming_parameters[name].grad,
                reference_parameters[name].grad,
                atol=1e-5,
                rtol=1e-4,
            ), name


def test_scnn_single_input_reducers_compatibility_unchanged():
    torch.manual_seed(107)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 7, 15)
    mask = (torch.rand(7, 15) > 0.4)

    with torch.no_grad():
        feat = model.backbone(image)
        expected_sum = model.sum_head[1](model.sum_head[0](feat), mask=mask)
        expected_mean = model.mean_head[1](model.mean_head[0](feat), mask=mask)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_scnn_input_resolution_mask_for_input_resolution_reducer():
    torch.manual_seed(115)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 8, 10)
    mask = ((torch.arange(8)[:, None] + 2 * torch.arange(10)[None, :]) % 4 != 0).to(torch.float32)

    with torch.no_grad():
        feat = model.backbone(image)
        expected_sum = model.sum_head[1](model.sum_head[0](feat), mask=mask.to(torch.bool))
        expected_mean = model.mean_head[1](model.mean_head[0](feat), mask=mask.to(torch.bool))

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_scnn_downsampled_reducer_mask_matches_reduced_feature_map():
    torch.manual_seed(117)
    model = DownsampledReducerNet().eval()
    image = torch.randn(1, 3, 9, 11)
    mask = (torch.arange(5)[:, None] + torch.arange(6)[None, :]) % 2 == 0

    with torch.no_grad():
        expected = model(image, mask=mask)

    scnn = _make_streaming(model, tile_size=5)
    with torch.no_grad():
        streamed = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed, expected, atol=1e-5, rtol=1e-4)


def test_scnn_downsampled_reducer_resizes_input_domain_mask():
    torch.manual_seed(118)
    model = DownsampledReducerNet(mask_resize=True).eval()
    image = torch.randn(1, 3, 9, 11)
    mask = (torch.arange(9)[:, None] + torch.arange(11)[None, :]) % 3 != 0

    with torch.no_grad():
        features = model.down(image)
        resized_mask = torch.nn.functional.interpolate(
            mask[None, None].to(torch.float32),
            size=features.shape[-2:],
            mode="nearest",
        ).to(torch.bool)[0, 0]
        expected = model.reducer(features, mask=resized_mask)

    scnn = _make_streaming(model, tile_size=5)
    with torch.no_grad():
        streamed = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    ("source_shape", "target_shape"),
    [
        ((9, 13), (4, 6)),       # downsample, non-integer ratio
        ((3, 5), (8, 12)),       # upsample, non-integer ratio
        ((8, 12), (4, 3)),       # exact integer ratios
        ((7, 11), (13, 17)),     # odd dimensions
        ((1, 9), (7, 1)),        # one-pixel source/target dimensions
        ((9, 1), (1, 8)),
    ],
)
def test_resize_nearest_bool_mask_matches_interpolate_for_all_chunk_sizes(source_shape, target_shape):
    source = (torch.arange(source_shape[0] * source_shape[1]).reshape(source_shape) % 3 == 0)
    expected = torch.nn.functional.interpolate(
        source[None, None].float(), size=target_shape, mode="nearest"
    )[0, 0].bool()

    results = [
        _resize_nearest_bool_mask(source, *target_shape, rows_per_chunk=chunk_size)
        for chunk_size in (1, 4, target_shape[0] + 3)
    ]
    assert all(torch.equal(result, expected) for result in results)
    assert all(torch.equal(result, results[0]) for result in results[1:])


def test_resize_nearest_bool_mask_chunked_path_never_uses_float_interpolation(monkeypatch):
    source = torch.rand(17, 19) > 0.5

    def forbidden_interpolate(*args, **kwargs):
        raise AssertionError("boolean mask resize must not call float interpolation")

    monkeypatch.setattr(torch.nn.functional, "interpolate", forbidden_interpolate)
    resized = _resize_nearest_bool_mask(source, 31, 37, rows_per_chunk=3)
    assert resized.shape == (31, 37)
    assert resized.dtype == torch.bool


def test_scnn_multi_head_masks_are_resized_in_each_reducer_output_domain():
    torch.manual_seed(120)
    model = MultiResolutionReducerNet().eval()
    image = torch.randn(1, 3, 17, 19)
    mask = (torch.arange(17)[:, None] + 2 * torch.arange(19)[None, :]) % 5 != 0
    scnn = _make_streaming(model, tile_size=9)

    with torch.no_grad():
        scnn.forward(image, mask=mask)

    assert len(scnn._prepared_reducer_domain_masks) == 2
    for head_idx, reducer in scnn._reducer_head_map.items():
        prepared = scnn._prepared_reducer_domain_masks[(id(reducer), head_idx)]
        target_shape = (
            scnn._current_output_heights[head_idx],
            scnn._current_output_widths[head_idx],
        )
        expected = torch.nn.functional.interpolate(
            mask[None, None].float(), size=target_shape, mode="nearest"
        )[0, 0].bool()
        assert prepared.shape == target_shape
        assert torch.equal(prepared.cpu(), expected)


def test_scnn_too_small_reducer_mask_fails_at_reducer_slice_site():
    torch.manual_seed(119)
    model = DownsampledReducerNet().eval()
    image = torch.randn(1, 3, 9, 11)
    mask = torch.ones((4, 6), dtype=torch.bool)

    scnn = _make_streaming(model, tile_size=5)
    with pytest.raises(ValueError, match="reducer/reduced feature spatial domain"):
        with torch.no_grad():
            scnn.forward(image, mask=mask)


def test_scnn_multi_input_reducer_failure_on_shape_mismatch():
    torch.manual_seed(109)
    model = ValueLogitsHeadNet().eval()
    image = torch.randn(1, 3, 9, 9)
    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        _ = scnn.forward(image)

    scnn._reducer_input_indices = {0: (0, 1)}
    bad_outputs = (torch.randn(1, 3, 3, 3), torch.randn(1, 3, 2, 3), torch.randn(1, 3, 3, 3))
    with pytest.raises(RuntimeError, match="tile input spatial mismatch"):
        scnn._stitch_forward_outputs([None, None, None], bad_outputs, 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), None)


def test_scnn_multi_input_reducer_failure_on_input_order_mismatch():
    torch.manual_seed(113)
    model = ValueLogitsHeadNet().eval()
    scnn = _make_streaming(model, tile_size=4)
    scnn._reducer_head_map = {0: scnn.stream_module.reducer}
    scnn._reducer_input_indices = {0: (0, 2, 1)}
    with pytest.raises(RuntimeError, match="input index order mismatch"):
        scnn._stitch_forward_outputs([None, None, None], (torch.randn(1, 3, 3, 3),) * 3, 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), None)


def test_streaming_fused_attention_gem_reducer_exposes_two_tensor_payload():
    torch.manual_seed(127)
    reducer = FusedAttentionGeMReducer(value_weights=(0.2, 0.5, 0.3)).to_streaming()
    y1 = torch.rand(1, 2, 4, 5) + 0.1
    y2 = torch.rand(1, 2, 4, 5) + 0.1
    y3 = torch.rand(1, 2, 4, 5) + 0.1
    logits = [torch.randn(1, 1, 4, 5) for _ in range(3)]

    base_reducer = FusedAttentionGeMReducer(value_weights=(0.2, 0.5, 0.3))
    base_reducer._streaming_passthrough = True
    base_payload = base_reducer(y1, y2, y3, *logits)
    payload = reducer(y1, y2, y3, *logits)

    assert len(base_payload) == 2
    assert torch.allclose(base_payload[0], 0.2 * y1 + 0.5 * y2 + 0.3 * y3)
    assert base_payload[1].shape == (1, 3, 4, 5)
    assert torch.allclose(base_payload[1], torch.cat(logits, dim=1))
    assert isinstance(reducer, StreamingFusedAttentionGeMReducer)
    assert len(payload) == 2
    expected_fused = 0.2 * y1 + 0.5 * y2 + 0.3 * y3
    assert torch.allclose(payload[0], expected_fused)
    assert payload[1].shape == (1, 3, 4, 5)
    assert torch.allclose(payload[1], torch.cat(logits, dim=1))
    assert reducer._last_inputs is payload
    assert reducer._last_output is payload[0]


def test_scnn_fused_attention_gem_internal_payload_count_shrinks_to_two():
    torch.manual_seed(131)
    model = FusedAttentionGeMHeadNet().eval()
    image = torch.rand(1, 3, 9, 11) + 0.05
    mask = torch.rand(9, 11) > 0.2

    with torch.no_grad():
        expected = model(image, mask=mask)

    scnn = _make_streaming(model, tile_size=4)
    assert isinstance(scnn.stream_module.reducer, StreamingFusedAttentionGeMReducer)
    assert len(scnn._tile_output_shapes) == 2

    with torch.no_grad():
        streamed = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed, expected, atol=1e-5, rtol=1e-4)


def _assert_non_edge_starts_match_alignment(scnn: StreamingCNN) -> None:
    align_h, align_w = scnn._compute_internal_alignment()
    assert align_h > 1
    assert align_w > 1
    for input_y, input_x, sides in scnn._last_forward_tiles:
        if not sides.bottom:
            assert input_y % align_h == 0
        if not sides.right:
            assert input_x % align_w == 0


def test_internal_alignment_includes_pools_and_streaming_convs_for_testnet_segment_model():
    torch.manual_seed(123)
    model = StreamingTestNet.create_model().eval()
    scnn = StreamingCNN(
        model,
        tile_shape=(1, 3, 128, 128),
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=True,
        normalize_on_gpu=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
    )

    alignment = scnn._compute_internal_alignment()
    assert alignment == (8, 8)

    valid_output_heights, valid_output_widths = scnn._compute_valid_output_sizes()
    valid_input_height, valid_input_width = scnn._compute_valid_input_step(
        valid_output_heights,
        valid_output_widths,
    )
    assert valid_input_height % alignment[0] == 0
    assert valid_input_width % alignment[1] == 0

    image = torch.rand(1, 3, 224, 224)
    scnn.forward(image)
    _assert_non_edge_starts_match_alignment(scnn)


class ResNetUpsamplingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18

        resnet = resnet18(weights=None)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def test_internal_alignment_keeps_resnet_encoder_upsampling_decoder_tile_phase():
    torch.manual_seed(456)
    model = ResNetUpsamplingDecoder().eval()
    scnn = StreamingCNN(
        model,
        tile_shape=(1, 3, 128, 128),
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=True,
        normalize_on_gpu=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
    )

    alignment = scnn._compute_internal_alignment()
    assert alignment[0] >= 8
    assert alignment[1] >= 8

    valid_output_heights, valid_output_widths = scnn._compute_valid_output_sizes()
    valid_input_height, valid_input_width = scnn._compute_valid_input_step(
        valid_output_heights,
        valid_output_widths,
    )
    assert valid_input_height % alignment[0] == 0
    assert valid_input_width % alignment[1] == 0

    image = torch.rand(1, 3, 224, 224)
    scnn.forward(image)
    _assert_non_edge_starts_match_alignment(scnn)


def test_single_head_backward_tile_step_rounds_down_to_internal_alignment():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.tile_gradient_lost = Lost(left=78, top=78, right=78, bottom=78)
    scnn.output_stride = torch.tensor([1, 1, 1])
    scnn._tile_output_shape = (1, 1, 1920, 1920)
    scnn._compute_internal_alignment = lambda: (8, 8)

    image = torch.empty(1, 3, 4000, 1920)
    grad_tensors = [torch.empty(1, 1, 4000, 1920)]

    tile_iter = scnn._prepare_backward_tile_iter_single_head(
        image=image,
        grad_tensors=grad_tensors,
        tile_height=1920,
        tile_width=1920,
    )

    y_starts = [input_y for input_y, _input_x, _sides in tile_iter]
    assert y_starts[:2] == [0, 1760]
    assert 1764 not in y_starts
    assert all(input_y % 8 == 0 and input_x % 8 == 0 for input_y, input_x, _sides in tile_iter)


def test_single_head_backward_alignment_debug_allows_edge_snapped_tiles(caplog):
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.tile_gradient_lost = Lost(left=78, top=78, right=78, bottom=78)
    scnn.output_stride = torch.tensor([1, 1, 1])
    scnn._tile_output_shape = (1, 1, 1920, 1920)
    scnn._compute_internal_alignment = lambda: (8, 8)
    scnn.debug_backward_tile_alignment = True

    image = torch.empty(1, 3, 4022, 1920)
    grad_tensors = [torch.empty(1, 1, 4022, 1920)]

    with caplog.at_level(logging.DEBUG, logger="lightstream.core.scnn.scnn"):
        tile_iter = scnn._prepare_backward_tile_iter_single_head(
            image=image,
            grad_tensors=grad_tensors,
            tile_height=1920,
            tile_width=1920,
        )

    bottom_tile_y, _bottom_tile_x, bottom_tile_sides = tile_iter[-1]
    assert bottom_tile_sides.bottom
    assert bottom_tile_y % 8 != 0
    assert "valid_grad_height=1760" in caplog.text
    assert f"y={bottom_tile_y}" in caplog.text
