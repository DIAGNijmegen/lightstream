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


def test_streaming_upsample_rejects_align_corners_true_for_bilinear():
    with pytest.raises(ValueError):
        StreamingUpsample2d(scale_factor=2.0, mode="bilinear", align_corners=True)


def test_constructor_keeps_upsample_modules():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.Upsample(scale_factor=2.0, mode="nearest"))
    constructor = StreamingConstructor(model, tile_size=32, verbose=False, statistics_on_cpu=True)
    assert torch.nn.Upsample in constructor.keep_modules


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


def test_bilinear_upsample_backward_drops_incomplete_support_cells(caplog):
    module = StreamingUpsample2d(scale_factor=2, mode="bilinear", align_corners=False)
    module.grad_lost = Lost(top=0, left=0, bottom=1, right=1)
    module.input_loc = Box(0, 0, 0, 0, Sides(left=0, top=0, right=0, bottom=0))

    x = torch.ones(1, 1, 3, 3, requires_grad=True)

    with caplog.at_level("DEBUG", logger="lightstream.core.scnn.streamingupsample"):
        module(x).sum().backward()

    expected = torch.tensor(
        [[[[4.0, 4.0, 0.0], [4.0, 4.0, 0.0], [0.0, 0.0, 0.0]]]]
    )
    torch.testing.assert_close(x.grad, expected)
    assert module.seen_indices == Box(0, 2, 2, 0, None)
    assert "dropped candidate low-res cells with incomplete high-res support" in caplog.text


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
