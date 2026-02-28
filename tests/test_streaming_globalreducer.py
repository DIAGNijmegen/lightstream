import torch

from lightstream.core.scnn.utils import Box, Lost, Sides
from lightstream.models.segment.globalreducer import GlobalReducer, StreamingGlobalReducer


def _iter_tiles(height: int, width: int, tile_h: int, tile_w: int):
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            sides = Sides(
                left=(x == 0),
                top=(y == 0),
                right=(x + tile_w >= width),
                bottom=(y + tile_h >= height),
            )
            yield y, x, sides


def test_streaming_global_reducer_matches_dense_forward_and_backward():
    r = 4.0
    eps = 1e-12
    x = torch.rand(1, 3, 4, 4, dtype=torch.float64, requires_grad=True)

    dense = GlobalReducer(r=r, eps=eps)
    y_dense = dense(x)
    y_dense.sum().backward()
    dense_grad = x.grad.detach().clone()

    streaming = StreamingGlobalReducer(r=r, eps=eps)
    streaming.output_stride = torch.tensor([1, 1, 1])

    tile_h = tile_w = 2
    last_y = None
    for y, x0, sides in _iter_tiles(4, 4, tile_h, tile_w):
        tile = x.detach()[:, :, y : y + tile_h, x0 : x0 + tile_w]
        streaming.input_loc = Box(y, tile_h, x0, tile_w, sides)
        last_y = streaming(tile)

    streaming.finalize_forward_state()
    assert last_y is not None
    assert torch.allclose(last_y, y_dense.detach(), atol=1e-10, rtol=1e-8)

    streaming.reset(keep_backward_state=True)
    stream_grad = torch.zeros_like(x.detach())
    for y, x0, sides in _iter_tiles(4, 4, tile_h, tile_w):
        tile = x.detach()[:, :, y : y + tile_h, x0 : x0 + tile_w].clone().requires_grad_(True)
        streaming.input_loc = Box(y, tile_h, x0, tile_w, sides)
        y_tile = streaming(tile)
        y_tile.sum().backward()
        stream_grad[:, :, y : y + tile_h, x0 : x0 + tile_w] = tile.grad

    assert torch.allclose(stream_grad, dense_grad, atol=1e-10, rtol=1e-8)


def test_reducer_only_scheduler_uses_effective_predecessor_lost_and_stride():
    from lightstream.core.scnn.scnn import StreamingCNN

    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn.tile_shape = (1, 1, 2560, 2560)
    scnn.tile_gradient_lost = Lost(0, 0, 0, 0)
    scnn._tile_output_shapes = [(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)]
    scnn._output_is_global_reducer = [True, True, True]
    scnn._output_stride_per_output = [
        torch.tensor([1, 1, 1]),
        torch.tensor([1, 1, 1]),
        torch.tensor([1, 1, 1]),
    ]
    scnn._output_effective_stride = [
        torch.tensor([1, 16, 16]),
        torch.tensor([1, 8, 8]),
        torch.tensor([1, 4, 4]),
    ]
    scnn._output_effective_lost = [
        Lost(120, 120, 104, 104),
        Lost(60, 60, 52, 52),
        Lost(30, 30, 26, 26),
    ]
    scnn._compute_internal_safe_input_step = lambda: (4000, 4000)
    scnn._compute_internal_alignment = lambda: (16, 16)

    h, w = StreamingCNN._compute_multi_output_input_step(scnn, [1, 1, 1], [1, 1, 1], include_grad_safe=True)

    assert h == 2336
    assert w == 2336
