import torch

from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.core.scnn.reduction import StreamingReductionHint


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


class TinyMultiHeadMixed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.reduce_head_0 = StreamingReductionHint("mean", tag="head_0")
        self.reduce_head_1 = StreamingReductionHint("none", tag="head_1")

    def forward(self, x):
        y = self.conv(x)
        return self.reduce_head_0(y), self.reduce_head_1(y[:, :2])


class TinySingleHeadReduced(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)
        self.reduce = StreamingReductionHint("sum", tag="main")

    def forward(self, x):
        return self.reduce(self.net(x))


def _build_streaming(module: torch.nn.Module) -> StreamingCNN:
    module.eval()
    return StreamingCNN(
        module,
        tile_shape=(1, 3, 8, 8),
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=True,
        normalize_on_gpu=False,
    )


def test_forward_without_hints_matches_non_streaming():
    torch.manual_seed(0)
    model = TinyBackbone().eval()
    image = torch.randn(1, 3, 13, 11)

    streaming = _build_streaming(model)
    streamed = streaming(image)

    with torch.no_grad():
        expected = model(image)

    torch.testing.assert_close(streamed, expected, atol=1e-5, rtol=1e-5)


def test_forward_with_sum_hint_matches_non_streaming_sum():
    torch.manual_seed(1)
    model = TinySingleHeadReduced().eval()
    image = torch.randn(1, 3, 15, 12)

    with torch.no_grad():
        expected_map = model.net(image)
        expected_sum = expected_map.sum(dim=(-2, -1), keepdim=True)

    streaming_sum = _build_streaming(model)
    out_sum = streaming_sum(image)

    assert out_sum.shape[-2:] == (1, 1)
    torch.testing.assert_close(out_sum, expected_sum, atol=1e-5, rtol=1e-5)


def test_mixed_reduction_modes_with_multi_output_structure():
    torch.manual_seed(2)
    model = TinyMultiHeadMixed().eval()
    image = torch.randn(1, 3, 14, 10)

    streaming = _build_streaming(model)
    out1, out2 = streaming(image)

    with torch.no_grad():
        y = model.conv(image)
        exp1 = y.mean(dim=(-2, -1), keepdim=True)
        exp2 = y[:, :2]

    assert out1.shape[-2:] == (1, 1)
    assert out2.shape[-2:] != (1, 1)
    torch.testing.assert_close(out1, exp1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out2, exp2, atol=1e-5, rtol=1e-5)


def test_backward_raises_if_any_output_is_reduced():
    model = TinyMultiHeadMixed().eval()
    image = torch.randn(1, 3, 13, 11)

    streaming = _build_streaming(model)
    grad = (torch.ones(1, 4, 1, 1), torch.ones(1, 2, 13, 11))

    try:
        streaming.backward(image, grad)
        raised = False
    except NotImplementedError:
        raised = True

    assert raised
