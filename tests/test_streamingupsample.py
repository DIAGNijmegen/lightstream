import pytest
import torch

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.core.scnn.streamingupsample import StreamingUpsample2d
from lightstream.core.scnn.utils import Box, Lost, Sides


def test_streaming_upsample_from_torch_matches_interpolate():
    upsample = torch.nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
    module = StreamingUpsample2d.from_torch_upsample(upsample)

    x = torch.rand(1, 3, 9, 11)
    expected = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    out = module(x)

    torch.testing.assert_close(out, expected)


def test_streaming_upsample_nearest_matches_interpolate():
    module = StreamingUpsample2d(scale_factor=2, mode="nearest")

    x = torch.rand(1, 3, 9, 11)
    expected = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    out = module(x)

    torch.testing.assert_close(out, expected)


def test_streaming_upsample_from_torch_nearest_matches_interpolate():
    upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
    module = StreamingUpsample2d.from_torch_upsample(upsample)

    x = torch.rand(1, 3, 9, 11)
    expected = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    out = module(x)

    assert module.align_corners is None
    torch.testing.assert_close(out, expected)


def test_streaming_upsample_to_torch_nearest_emits_valid_upsample():
    module = StreamingUpsample2d(scale_factor=2, mode="nearest")
    upsample = module.to_torch_upsample()

    x = torch.rand(1, 3, 9, 11)
    expected = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    out = upsample(x)

    assert upsample.align_corners is None
    torch.testing.assert_close(out, expected)


def test_streaming_upsample_rejects_align_corners_for_nearest():
    with pytest.raises(ValueError, match="align_corners=None"):
        StreamingUpsample2d(scale_factor=2, mode="nearest", align_corners=True)


def test_streaming_upsample_rejects_align_corners_true_for_bilinear():
    with pytest.raises(ValueError):
        StreamingUpsample2d(scale_factor=2.0, mode="bilinear", align_corners=True)


def test_constructor_keeps_upsample_modules():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.Upsample(scale_factor=2.0, mode="nearest"))
    constructor = StreamingConstructor(model, tile_size=32, verbose=False, statistics_on_cpu=True)
    assert torch.nn.Upsample in constructor.keep_modules


def test_constructor_converts_nearest_upsample_to_streaming_upsample():
    upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), upsample).eval()
    constructor = StreamingConstructor(
        model,
        tile_size=8,
        verbose=False,
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
    )

    scnn = constructor.prepare_streaming_model()
    converted = scnn.stream_module[1]

    assert isinstance(converted, StreamingUpsample2d)
    assert converted.mode == "nearest"
    assert converted.scale_factor == upsample.scale_factor
    assert converted.size == upsample.size
    assert converted.recompute_scale_factor == upsample.recompute_scale_factor
    assert converted.align_corners is None


def test_bilinear_upsample_statistics_add_explicit_border_loss():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.eps = 1e-5
    scnn.dtype = torch.float32
    scnn.device = torch.device("cpu")
    scnn._saved_tensors = {}
    scnn._module_stats = {}
    scnn._print_verbose = lambda *args, **kwargs: None

    module = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    inpt = torch.ones(1, 3, 5, 7)

    with torch.no_grad():
        output = module(inpt)
        scnn._forward_gather_statistics_hook(module, (inpt,), output)

    lost = scnn._module_stats[module]["lost"]
    assert (lost.top, lost.bottom, lost.left, lost.right) == (1, 1, 1, 1)
    assert torch.all(output[:, :, :1, :] == 0)
    assert torch.all(output[:, :, -1:, :] == 0)
    assert torch.all(output[:, :, :, :1] == 0)
    assert torch.all(output[:, :, :, -1:] == 0)
    assert torch.all(output[:, :, 1:-1, 1:-1] == 1)


def test_nearest_upsample_statistics_do_not_add_explicit_border_loss():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.eps = 1e-5
    scnn.dtype = torch.float32
    scnn.device = torch.device("cpu")
    scnn._saved_tensors = {}
    scnn._module_stats = {}
    scnn._print_verbose = lambda *args, **kwargs: None

    module = torch.nn.Upsample(scale_factor=2, mode="nearest")
    inpt = torch.ones(1, 3, 5, 7)

    with torch.no_grad():
        output = module(inpt)
        scnn._forward_gather_statistics_hook(module, (inpt,), output)

    lost = scnn._module_stats[module]["lost"]
    assert lost == Lost(0, 0, 0, 0)
    assert output.shape[-2:] == (10, 14)
    assert torch.all(output == 1)


@pytest.mark.parametrize(
    ("scale_factor", "expected_loss"),
    [
        (2, 1),
        (4, 2),
        (8, 4),
    ],
)
def test_bilinear_upsample_statistics_scale_factor_loss_formula(scale_factor, expected_loss):
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.eps = 1e-5
    scnn.dtype = torch.float32
    scnn.device = torch.device("cpu")
    scnn._saved_tensors = {}
    scnn._module_stats = {}
    scnn._print_verbose = lambda *args, **kwargs: None

    module = torch.nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
    inpt = torch.ones(1, 3, 5, 5)

    with torch.no_grad():
        output = module(inpt)
        scnn._forward_gather_statistics_hook(module, (inpt,), output)

    lost = scnn._module_stats[module]["lost"]
    assert (lost.top, lost.bottom, lost.left, lost.right) == (
        expected_loss,
        expected_loss,
        expected_loss,
        expected_loss,
    )


def test_streaming_upsample_backward_keeps_pre_upsample_coordinate_gradients():
    module = StreamingUpsample2d(scale_factor=2, mode="nearest")
    module.output_stride = torch.tensor([1, 1, 1])
    module.pre_upsample_output_stride = torch.tensor([1, 2, 2])

    first = torch.ones(1, 1, 4, 4, requires_grad=True)
    module.input_loc = Box(0, 0, 0, 0, Sides(left=0, top=0, right=0, bottom=0))
    module(first).sum().backward()
    assert torch.count_nonzero(first.grad).item() == first.numel()

    second = torch.ones(1, 1, 4, 4, requires_grad=True)
    module.input_loc = Box(0, 0, 4, 0, Sides(left=0, top=0, right=0, bottom=0))
    module(second).sum().backward()

    assert torch.count_nonzero(second.grad).item() == second.numel()


def test_streaming_upsample_backward_masks_input_gradient_with_backward_valid_lost():
    module = StreamingUpsample2d(scale_factor=2, mode="nearest")
    module.backward_valid_lost = Lost(top=1, left=1, bottom=1, right=1)
    module.input_loc = Box(0, 0, 0, 0, Sides(left=0, top=0, right=0, bottom=0))

    x = torch.ones(1, 1, 4, 4, requires_grad=True)
    module(x).sum().backward()

    expected = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 4.0, 0.0], [0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]]
    )
    torch.testing.assert_close(x.grad, expected)


def test_streaming_upsample_backward_valid_lost_is_side_aware():
    module = StreamingUpsample2d(scale_factor=2, mode="nearest")
    module.backward_valid_lost = Lost(top=1, left=1, bottom=1, right=1)
    module.input_loc = Box(0, 0, 0, 0, Sides(left=1, top=1, right=0, bottom=0))

    x = torch.ones(1, 1, 4, 4, requires_grad=True)
    module(x).sum().backward()

    expected = torch.tensor(
        [[[[4.0, 4.0, 4.0, 0.0], [4.0, 4.0, 4.0, 0.0], [4.0, 4.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]]
    )
    torch.testing.assert_close(x.grad, expected)


def test_streaming_upsample_backward_valid_lost_does_not_depend_on_seen_indices_ownership():
    module = StreamingUpsample2d(scale_factor=2, mode="nearest")
    module.backward_valid_lost = Lost(top=1, left=1, bottom=1, right=1)
    module.pre_upsample_output_stride = torch.tensor([1, 2, 2])
    module.seen_indices = Box(0, 4, 4, 0, None)
    module.input_loc = Box(0, 0, 0, 0, Sides(left=0, top=0, right=0, bottom=0))

    x = torch.ones(1, 1, 4, 4, requires_grad=True)
    module(x).sum().backward()

    assert torch.count_nonzero(x.grad).item() == 4
    torch.testing.assert_close(x.grad[:, :, 1:3, 1:3], torch.full((1, 1, 2, 2), 4.0))


def test_bilinear_upsample_backward_uses_stat_derived_lowres_lost_region():
    module = StreamingUpsample2d(scale_factor=2, mode="bilinear", align_corners=False)
    # This high-resolution grad-output loss is intentionally different from the
    # low-resolution backward-input loss. The backward pass must not use it to
    # crop grad_in directly.
    module.grad_lost = Lost(top=0, left=0, bottom=99, right=99)
    module.upsample_backward_input_lost = Lost(top=1, left=1, bottom=1, right=1)
    module.input_loc = Box(0, 0, 0, 0, Sides(left=0, top=0, right=0, bottom=0))

    x = torch.ones(1, 1, 4, 4, requires_grad=True)
    module(x).sum().backward()

    expected = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 4.0, 0.0], [0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]]
    )
    torch.testing.assert_close(x.grad, expected)
    assert module.seen_indices == Box(0, 0, 0, 0, None)


def test_upsample_statistics_store_pre_upsample_output_stride():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.eps = 1e-5
    scnn.dtype = torch.float32
    scnn.device = torch.device("cpu")
    scnn._saved_tensors = {}
    scnn._module_stats = {}
    scnn._stats_per_grad_fn = {}
    scnn._print_verbose = lambda *args, **kwargs: None

    module = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    inpt = torch.ones(1, 3, 5, 5, requires_grad=True)
    output = module(inpt)

    with torch.no_grad():
        scnn._forward_gather_statistics_hook(module, (inpt,), output.detach().clone())
    scnn._forward_gather_statistics_hook(module, (inpt,), output)

    stats = scnn._module_stats[module]
    assert stats["output_stride"].tolist() == [1, 1, 1]
    assert stats["pre_upsample_output_stride"].tolist() == [1, 2, 2]
    assert stats["backward_valid_lost"] == Lost(0, 0, 0, 0)


def test_upsample_backward_statistics_store_backward_valid_lost():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.eps = 1e-5
    scnn.dtype = torch.float32
    scnn.device = torch.device("cpu")
    scnn._saved_tensors = {}
    scnn._module_stats = {}
    scnn._print_verbose = lambda *args, **kwargs: None

    module = torch.nn.Upsample(scale_factor=2, mode="nearest")
    inpt = torch.ones(1, 1, 4, 4, requires_grad=True)
    output = module(inpt)
    scnn._module_stats[module] = {}

    output.sum().backward()
    scnn._backward_gather_statistics_hook(module, (inpt.grad,), (torch.ones_like(output),))

    assert scnn._module_stats[module]["backward_valid_lost"] == Lost(0, 0, 0, 0)


def test_safe_input_step_accounts_for_upsample_forward_and_backward_lost_regions():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.tile_shape = (1, 1, 64, 64)
    scnn.tile_gradient_lost = Lost(0, 0, 0, 0)
    scnn._print_verbose = lambda *args, **kwargs: None

    module = StreamingUpsample2d(scale_factor=2, mode="bilinear", align_corners=False)
    module.grad_lost = Lost(top=3, left=5, bottom=7, right=11)
    module.upsample_backward_input_lost = Lost(top=2, left=4, bottom=6, right=8)
    module.pre_upsample_output_stride = torch.tensor([1, 4, 4])
    module.output_stride = torch.tensor([1, 2, 2])
    module.post_upsample_output_stride = module.output_stride
    scnn.stream_module = torch.nn.Sequential(module)
    scnn._module_stats = {module: {"lost": Lost(top=3, left=5, bottom=7, right=11)}}

    safe_h, safe_w = scnn._compute_internal_safe_input_step()

    # Forward loss is in post-upsample output coordinates:
    #   H: 64 - (3 + 7) * 2 = 44
    #   W: 64 - (5 + 11) * 2 = 32
    # Backward input loss is in pre-upsample low-resolution coordinates:
    #   H: 64 - (2 + 6) * 4 = 32
    #   W: 64 - (4 + 8) * 4 = 16
    assert (safe_h, safe_w) == (32, 16)


def test_single_output_valid_input_step_is_reduced_by_upsample_safe_step_without_losing_alignment():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.tile_shape = (1, 1, 64, 64)
    scnn.tile_gradient_lost = Lost(0, 0, 0, 0)
    scnn._tile_output_shapes = [(1, 1, 32, 32)]
    scnn._output_stride_per_output = [torch.tensor([1, 2, 2])]
    scnn._print_verbose = lambda *args, **kwargs: None

    module = StreamingUpsample2d(scale_factor=2, mode="bilinear", align_corners=False)
    module.grad_lost = Lost(top=0, left=0, bottom=0, right=0)
    module.upsample_backward_input_lost = Lost(top=3, left=3, bottom=3, right=3)
    module.pre_upsample_output_stride = torch.tensor([1, 4, 4])
    module.output_stride = torch.tensor([1, 2, 2])
    module.post_upsample_output_stride = module.output_stride
    scnn.stream_module = torch.nn.Sequential(module)
    scnn._module_stats = {module: {"lost": Lost(top=0, left=0, bottom=0, right=0)}}

    step_h, step_w = scnn._compute_valid_input_step([32], [32])

    assert (step_h, step_w) == (40, 40)
