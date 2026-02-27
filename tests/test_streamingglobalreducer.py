import torch

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.scnn.streamingglobalreducer import StreamingGlobalReducer
from lightstream.models.segment.globalreducer import GlobalReducer


def test_streaming_global_reducer_forward_matches_non_streaming():
    torch.manual_seed(0)
    x = torch.rand(2, 4, 11, 13)

    reducer = GlobalReducer(r=4.0, eps=1e-12)
    streaming_reducer = StreamingGlobalReducer(r=4.0, eps=1e-12)

    expected = reducer(x)
    out = streaming_reducer(x)

    torch.testing.assert_close(out, expected)


def test_streaming_global_reducer_backward_matches_non_streaming():
    torch.manual_seed(0)
    x_stream = torch.rand(2, 3, 7, 9, requires_grad=True)
    x_normal = x_stream.detach().clone().requires_grad_(True)

    reducer = GlobalReducer(r=4.0, eps=1e-12)
    streaming_reducer = StreamingGlobalReducer(r=4.0, eps=1e-12)

    grad_out = torch.rand(2, 3, 1, 1)

    out_stream = streaming_reducer(x_stream)
    out_normal = reducer(x_normal)

    out_stream.backward(grad_out)
    out_normal.backward(grad_out)

    torch.testing.assert_close(out_stream.detach(), out_normal.detach())
    torch.testing.assert_close(x_stream.grad, x_normal.grad, rtol=1e-5, atol=1e-7)


def test_constructor_keeps_global_reducer_modules():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), GlobalReducer())
    constructor = StreamingConstructor(model, tile_size=32, verbose=False, statistics_on_cpu=True)
    assert GlobalReducer in constructor.keep_modules
