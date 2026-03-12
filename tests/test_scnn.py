import torch
import torch.nn as nn
import pytest

from lightstream.core.constructor import StreamingConstructor
from lightstream.modules.reducer import Reducer


class AllReducerHeadsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.sum_head = nn.Sequential(nn.Conv2d(6, 2, kernel_size=1, bias=False), Reducer(mode="sum"))
        self.mean_head = nn.Sequential(nn.Conv2d(6, 3, kernel_size=1, bias=False), Reducer(mode="mean"))

    def forward(self, x):
        feat = self.backbone(x)
        return self.sum_head(feat), self.mean_head(feat)


class MixedHeadsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.raw_head = nn.Conv2d(5, 4, kernel_size=1, bias=False)
        self.reducer_head = nn.Sequential(nn.Conv2d(5, 4, kernel_size=1, bias=False), Reducer(mode="mean"))

    def forward(self, x):
        feat = self.backbone(x)
        return {"raw": self.raw_head(feat), "reduced": self.reducer_head(feat)}


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
