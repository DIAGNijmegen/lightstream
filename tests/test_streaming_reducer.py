import torch
import torch.nn as nn

from lightstream.core.constructor import StreamingConstructor
from lightstream.modules.reducer import Reducer, StreamingReducer


class MixedReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.raw_head = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.mean_conv = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.mean_reducer = Reducer(mode="mean")

    def forward(self, x, mask=None):
        feat = self.backbone(x)
        reduced = self.mean_reducer(self.mean_conv(feat), mask=mask)
        return {"raw": self.raw_head(feat), "reduced": reduced}


class MultiReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.sum_conv = nn.Conv2d(5, 2, kernel_size=1, bias=False)
        self.sum_reducer = Reducer(mode="sum")
        self.mean_conv = nn.Conv2d(5, 2, kernel_size=1, bias=False)
        self.mean_reducer = Reducer(mode="mean")

    def forward(self, x, mask=None):
        feat = self.backbone(x)
        return self.sum_reducer(self.sum_conv(feat), mask=mask), self.mean_reducer(self.mean_conv(feat), mask=mask)


def _make_streaming(model: nn.Module, tile_size: int = 4):
    constructor = StreamingConstructor(
        model,
        tile_size=tile_size,
        saliency=True,
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
        expected = model(image, mask=None)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed = scnn.forward(image, mask=None)

    assert streamed["raw"].shape == expected["raw"].shape
    assert streamed["reduced"].shape == expected["reduced"].shape
    assert torch.allclose(streamed["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)


def test_streaming_reducer_multiple_reducers_modes_forward_parity():
    torch.manual_seed(2)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 8, 10)

    with torch.no_grad():
        expected_sum, expected_mean = model(image, mask=None)

    scnn = _make_streaming(model, tile_size=5)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image, mask=None)

    assert streamed_sum.shape == expected_sum.shape
    assert streamed_mean.shape == expected_mean.shape
    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_streaming_reducer_multiple_reducers_modes_forward_parity_with_mask():
    torch.manual_seed(12)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 8, 10)
    mask = torch.zeros((8, 10), dtype=torch.bool)
    mask[1:7, 2:8] = True

    with torch.no_grad():
        expected_sum, expected_mean = model(image, mask=mask)

    scnn = _make_streaming(model, tile_size=5)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image, mask=mask)

    assert streamed_sum.shape == expected_sum.shape
    assert streamed_mean.shape == expected_mean.shape
    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_streaming_reducer_backward_parity():
    torch.manual_seed(3)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 11, 9)
    mask = torch.rand((11, 9)) > 0.35

    reference = MultiReducerNet().eval()
    reference.load_state_dict(model.state_dict())

    ref_image = image.clone().requires_grad_(True)
    ref_sum, ref_mean = reference(ref_image, mask=mask)
    ref_loss = (0.7 * ref_sum).sum() + (1.3 * ref_mean).sum()
    ref_loss.backward()

    scnn = _make_streaming(model, tile_size=4)
    streamed_sum, streamed_mean = scnn.forward(image.clone(), mask=mask)
    grad_sum = torch.full_like(streamed_sum, 0.7)
    grad_mean = torch.full_like(streamed_mean, 1.3)
    scnn.backward(image.clone(), (grad_sum, grad_mean))

    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}

    for name, ref_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_grad, atol=1e-5, rtol=1e-4), name


def test_streaming_reducer_backward_parity_tiny_odd_image():
    torch.manual_seed(5)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 3, 5)

    reference = MultiReducerNet().eval()
    reference.load_state_dict(model.state_dict())

    ref_image = image.clone().requires_grad_(True)
    ref_sum, ref_mean = reference(ref_image, mask=None)
    ref_loss = (0.2 * ref_sum).sum() + (-0.4 * ref_mean).sum()
    ref_loss.backward()

    scnn = _make_streaming(model, tile_size=6)
    streamed_sum, streamed_mean = scnn.forward(image.clone(), mask=None)
    grad_sum = torch.full_like(streamed_sum, 0.2)
    grad_mean = torch.full_like(streamed_mean, -0.4)
    scnn.backward(image.clone(), (grad_sum, grad_mean))

    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}

    for name, ref_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_grad, atol=1e-5, rtol=1e-4), name


def test_streaming_reducer_backward_parity_with_mask():
    torch.manual_seed(13)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 11, 9)
    mask = torch.zeros((11, 9), dtype=torch.bool)
    mask[2:10, 1:8] = True

    reference = MultiReducerNet().eval()
    reference.load_state_dict(model.state_dict())

    ref_image = image.clone().requires_grad_(True)
    ref_sum, ref_mean = reference(ref_image, mask=mask)
    ref_loss = (0.7 * ref_sum).sum() + (1.3 * ref_mean).sum()
    ref_loss.backward()

    scnn = _make_streaming(model, tile_size=4)
    streamed_sum, streamed_mean = scnn.forward(image.clone(), mask=mask)
    grad_sum = torch.full_like(streamed_sum, 0.7)
    grad_mean = torch.full_like(streamed_mean, 1.3)
    scnn.backward(image.clone(), (grad_sum, grad_mean))

    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}

    for name, ref_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_grad, atol=1e-5, rtol=1e-4), name


def test_streaming_reducer_forward_parity_all_zero_mask_mean_clamped():
    torch.manual_seed(14)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 7, 9)
    mask = torch.zeros((7, 9), dtype=torch.bool)

    with torch.no_grad():
        expected_sum, expected_mean = model(image, mask=mask)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image, mask=mask)

    assert torch.allclose(expected_sum, torch.zeros_like(expected_sum))
    assert torch.allclose(expected_mean, torch.zeros_like(expected_mean))
    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_streaming_reducer_running_count_uses_fp32_accumulator():
    reducer = StreamingReducer(mode="mean")
    tile = torch.ones((1, 2, 2, 3), dtype=torch.float16)

    reducer.accumulate_tile(tile)

    assert reducer.running_sum.dtype == torch.float16
    assert reducer.running_count.dtype == torch.float32

    output = reducer.finalize_stream()
    assert output.dtype == tile.dtype
    assert torch.allclose(output, torch.ones((1, 2, 1, 1), dtype=tile.dtype))
