import torch

from lightstream.core.scnn.scnn import StreamingCNN


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 2, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class TinyMultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        y = self.conv(x)
        return y, y[:, :2]


def _build_streaming(module: torch.nn.Module, reduction_mode: str = "none") -> StreamingCNN:
    module.eval()
    return StreamingCNN(
        module,
        tile_shape=(1, 3, 8, 8),
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=True,
        normalize_on_gpu=False,
        reduction_mode=reduction_mode,
    )


def test_forward_none_matches_non_streaming():
    torch.manual_seed(0)
    model = TinyBackbone().eval()
    image = torch.randn(1, 3, 13, 11)

    streaming = _build_streaming(model, reduction_mode="none")
    streamed = streaming(image)

    with torch.no_grad():
        expected = model(image)

    torch.testing.assert_close(streamed, expected, atol=1e-5, rtol=1e-5)


def test_forward_sum_and_mean_match_non_streaming():
    torch.manual_seed(1)
    model = TinyBackbone().eval()
    image = torch.randn(1, 3, 15, 12)

    with torch.no_grad():
        expected_map = model(image)
        expected_sum = expected_map.sum(dim=(-2, -1), keepdim=True)
        expected_mean = expected_map.mean(dim=(-2, -1), keepdim=True)

    streaming_sum = _build_streaming(model, reduction_mode="sum")
    streaming_mean = _build_streaming(model, reduction_mode="mean")

    out_sum = streaming_sum(image)
    out_mean = streaming_mean(image)

    assert out_sum.shape[-2:] == (1, 1)
    assert out_mean.shape[-2:] == (1, 1)
    torch.testing.assert_close(out_sum, expected_sum, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_mean, expected_mean, atol=1e-5, rtol=1e-5)


def test_reduction_mode_works_with_multi_output_structure():
    torch.manual_seed(2)
    model = TinyMultiHead().eval()
    image = torch.randn(1, 3, 14, 10)

    streaming = _build_streaming(model, reduction_mode="mean")
    out1, out2 = streaming(image)

    with torch.no_grad():
        exp1, exp2 = model(image)
        exp1 = exp1.mean(dim=(-2, -1), keepdim=True)
        exp2 = exp2.mean(dim=(-2, -1), keepdim=True)

    torch.testing.assert_close(out1, exp1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out2, exp2, atol=1e-5, rtol=1e-5)


def test_backward_raises_for_reduction_modes():
    model = TinyBackbone().eval()
    image = torch.randn(1, 3, 13, 11)

    streaming = _build_streaming(model, reduction_mode="sum")
    grad = torch.ones(1, 2, 1, 1)

    try:
        streaming.backward(image, grad)
        raised = False
    except NotImplementedError:
        raised = True

    assert raised
