import torch
import torch.nn as nn

from lightstream.core.constructor import StreamingConstructor
from lightstream.modules.reducer import Reducer


class MixedReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.raw_head = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.mean_head = nn.Sequential(nn.Conv2d(4, 2, kernel_size=1, bias=False), Reducer(mode="mean"))

    def forward(self, x):
        feat = self.backbone(x)
        return {"raw": self.raw_head(feat), "reduced": self.mean_head(feat)}


class MultiReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.sum_head = nn.Sequential(nn.Conv2d(5, 2, kernel_size=1, bias=False), Reducer(mode="sum"))
        self.mean_head = nn.Sequential(nn.Conv2d(5, 2, kernel_size=1, bias=False), Reducer(mode="mean"))

    def forward(self, x):
        feat = self.backbone(x)
        return self.sum_head(feat), self.mean_head(feat)


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


def test_streaming_reducer_mixed_outputs_forward_parity():
    torch.manual_seed(1)
    model = MixedReducerNet().eval()
    image = torch.randn(1, 3, 9, 11)

    with torch.no_grad():
        expected = model(image)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed = scnn.forward(image)

    assert streamed["raw"].shape == expected["raw"].shape
    assert streamed["reduced"].shape == expected["reduced"].shape
    assert torch.allclose(streamed["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)


def test_streaming_reducer_multiple_reducers_modes_forward_parity():
    torch.manual_seed(2)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 8, 10)

    with torch.no_grad():
        expected_sum, expected_mean = model(image)

    scnn = _make_streaming(model, tile_size=5)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image)

    assert streamed_sum.shape == expected_sum.shape
    assert streamed_mean.shape == expected_mean.shape
    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)
