import torch
import torch.nn as nn

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.reducer import MeanReducer, StreamingMeanReducer, StreamingSumReducer, SumReducer


class MixedReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.raw_head = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.mean_head = nn.Sequential(nn.Conv2d(4, 2, kernel_size=1, bias=False), MeanReducer())

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
        self.sum_head = nn.Sequential(nn.Conv2d(5, 2, kernel_size=1, bias=False), SumReducer())
        self.mean_head = nn.Sequential(nn.Conv2d(5, 2, kernel_size=1, bias=False), MeanReducer())

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


def test_streaming_reducer_backward_parity():
    torch.manual_seed(3)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 11, 9)

    reference = MultiReducerNet().eval()
    reference.load_state_dict(model.state_dict())

    ref_image = image.clone().requires_grad_(True)
    ref_sum, ref_mean = reference(ref_image)
    ref_loss = (0.7 * ref_sum).sum() + (1.3 * ref_mean).sum()
    ref_loss.backward()

    scnn = _make_streaming(model, tile_size=4)
    streamed_sum, streamed_mean = scnn.forward(image.clone())
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
    ref_sum, ref_mean = reference(ref_image)
    ref_loss = (0.2 * ref_sum).sum() + (-0.4 * ref_mean).sum()
    ref_loss.backward()

    scnn = _make_streaming(model, tile_size=6)
    streamed_sum, streamed_mean = scnn.forward(image.clone())
    grad_sum = torch.full_like(streamed_sum, 0.2)
    grad_mean = torch.full_like(streamed_mean, -0.4)
    scnn.backward(image.clone(), (grad_sum, grad_mean))

    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}

    for name, ref_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_grad, atol=1e-5, rtol=1e-4), name


def test_streaming_mean_reducer_running_count_uses_fp32_accumulator():
    reducer = StreamingMeanReducer()
    tile = torch.ones((1, 2, 2, 3), dtype=torch.float16)

    reducer.accumulate_valid_tile(tile, valid_mask=torch.ones((2, 3), dtype=torch.bool))

    assert reducer.running_sum.dtype == torch.float16
    assert reducer.running_count.dtype == torch.float32

    output = reducer.finalize_stream()
    assert output.dtype == tile.dtype
    assert torch.allclose(output, torch.ones((1, 2, 1, 1), dtype=tile.dtype))


def test_streaming_sum_reducer_does_not_normalize():
    reducer = StreamingSumReducer()
    tile = torch.ones((1, 2, 2, 3), dtype=torch.float16)

    reducer.accumulate_valid_tile(tile, valid_mask=torch.ones((2, 3), dtype=torch.bool))

    output = reducer.finalize_stream()
    assert output.dtype == tile.dtype
    assert torch.allclose(output, torch.full((1, 2, 1, 1), 6.0, dtype=tile.dtype))

from lightstream.core.reducer import StreamingGeMReducer


def _gem_reference(x: torch.Tensor, mask: torch.Tensor, r: torch.Tensor, eps: float):
    x_clamped = x.clamp_min(eps)
    mask4 = mask.to(dtype=x.dtype, device=x.device)[None, None]
    n = mask4.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    m = (x_clamped.pow(r) * mask4).sum(dim=(-2, -1), keepdim=True) / n
    return m.clamp_min(eps).pow(1.0 / r)


def test_streaming_gem_default_r_init():
    reducer = StreamingGeMReducer()
    assert reducer.learnable_r is False
    assert torch.isclose(reducer.current_r, torch.tensor(4.0), atol=1e-6)


def test_streaming_gem_learnable_r_matches_init():
    reducer = StreamingGeMReducer(r_init=16.0, learnable_r=True)
    assert isinstance(reducer.r, torch.nn.Parameter)
    assert torch.isclose(reducer.current_r.detach(), torch.tensor(16.0), atol=1e-6)


def test_streaming_gem_forward_parity_masked_tiny_odd_shape():
    torch.manual_seed(7)
    reducer = StreamingGeMReducer(r_init=3.5, learnable_r=False, eps=1e-6)
    x = torch.rand(1, 2, 3, 5) + 0.01
    mask = torch.tensor([[1, 0, 1, 0, 1], [1, 1, 0, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.bool)

    reducer.start_stream(output_height=3, output_width=5, batch_size=1, channels=2, device=x.device, dtype=x.dtype)
    reducer.accumulate_stream_tile(x[:, :, :2, :3], 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,2,0,3), user_mask=mask[:2,:3])
    reducer.accumulate_stream_tile(x[:, :, :2, 3:], 0, 1, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,2,3,5), user_mask=mask[:2,3:])
    reducer.accumulate_stream_tile(x[:, :, 2:, :], 1, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (2,3,0,5), user_mask=mask[2:,:])
    y_stream = reducer.finish_stream()

    y_ref = _gem_reference(x, mask, reducer.current_r.to(dtype=x.dtype), reducer.eps)
    assert torch.allclose(y_stream, y_ref, atol=1e-5, rtol=1e-4)


def test_streaming_gem_backward_input_and_r_grad_parity():
    torch.manual_seed(9)
    x = (torch.rand(1, 3, 5, 7) + 0.05).requires_grad_(True)
    mask = (torch.rand(5, 7) > 0.3)

    reducer = StreamingGeMReducer(r_init=2.3, learnable_r=True, eps=1e-6)
    reducer.start_stream(output_height=5, output_width=7, batch_size=1, channels=3, device=x.device, dtype=x.dtype)
    reducer.accumulate_stream_tile(x[:, :, :, :4], 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,5,0,4), user_mask=mask[:, :4])
    reducer.accumulate_stream_tile(x[:, :, :, 4:], 0, 1, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,5,4,7), user_mask=mask[:, 4:])
    y_stream = reducer.finish_stream()
    loss_stream = y_stream.sum()
    loss_stream.backward()
    grad_x_stream = x.grad.detach().clone()
    grad_r_stream = reducer.r.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    r_ref = torch.nn.Parameter(reducer.r.detach().clone())
    y_ref = _gem_reference(x_ref, mask, r_ref.to(dtype=x_ref.dtype), reducer.eps)
    y_ref.sum().backward()

    assert torch.allclose(grad_x_stream, x_ref.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(grad_r_stream, r_ref.grad, atol=1e-5, rtol=1e-4)


def test_streaming_gem_fp16_stability_accumulator_fp32():
    torch.manual_seed(11)
    reducer = StreamingGeMReducer(r_init=4.0, learnable_r=False)
    x = (torch.rand(1, 2, 4, 4, dtype=torch.float16) + 0.01)
    mask = torch.ones((4, 4), dtype=torch.bool)

    reducer.start_stream(output_height=4, output_width=4, batch_size=1, channels=2, device=x.device, dtype=x.dtype)
    reducer.accumulate_stream_tile(x, 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,4,0,4), user_mask=mask)
    y = reducer.finish_stream()

    assert reducer.running_count.dtype == torch.float32
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()
