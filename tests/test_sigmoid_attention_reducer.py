from types import SimpleNamespace
import pytest
import torch
from lightstream.core.reducer import (
    SigmoidAttentionPoolingReducer,
    StreamingSigmoidAttentionPoolingReducer,
)


@pytest.mark.parametrize("stopgrad", [False, True])
@pytest.mark.parametrize("learnable", [False, True])
def test_formula_shape_mask_and_gradients(stopgrad, learnable):
    torch.manual_seed(8)
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 0, 1, 1], [1, 0, 1, 1, 0], [1, 1, 1, 0, 1], [0, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    reducer = SigmoidAttentionPoolingReducer(
        0.65, learnable, stopgrad, accumulator_dtype=torch.float64
    )
    y = reducer(x, mask=mask)
    scores = torch.sigmoid(x)
    if stopgrad:
        scores = scores.detach()
    q = (scores / reducer.current_tau).masked_fill(~mask, -torch.inf)
    expected = (torch.softmax(q.flatten(2), -1).view_as(x) * x).sum(
        (-2, -1), keepdim=True
    )
    assert y.shape == (2, 3, 1, 1)
    assert torch.allclose(y, expected)
    y.sum().backward()
    assert x.grad is not None
    assert (reducer.raw_tau.grad is not None) == learnable


@pytest.mark.parametrize("stopgrad", [False, True])
@pytest.mark.parametrize("learnable", [False, True])
def test_streaming_forward_and_backward_replay_parity(stopgrad, learnable):
    torch.manual_seed(11)
    offline = SigmoidAttentionPoolingReducer(
        0.8, learnable, stopgrad, accumulator_dtype=torch.float64
    )
    x = torch.randn(2, 4, 5, 7, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(2, 4, 1, 1, dtype=torch.float64)
    offline(x).backward(upstream)
    expected_x_grad = x.grad.clone()
    expected_tau_grad = None if not learnable else offline.raw_tau.grad.clone()
    streaming = offline.to_streaming()
    streaming.start_stream(5, 7, 2, 4, x.device, x.dtype)
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    tiles = []
    for y0, y1 in ((0, 3), (2, 5)):
        for x0, x1 in ((0, 4), (3, 7)):
            tile = x.detach()[..., y0:y1, x0:x1]
            streaming.accumulate_stream_tile(tile, y0, x0, sides, (y0, y1, x0, x1))
            tiles.append((y0, y1, x0, x1, tile))
    assert torch.allclose(streaming.finish_stream(), offline(x.detach()), atol=1e-12)
    seen = torch.zeros(5, 7, dtype=torch.bool)
    replay_grad = torch.zeros_like(x)
    for y0, y1, x0, x1, tile in tiles:
        valid = ~seen[y0:y1, x0:x1]
        seen[y0:y1, x0:x1] = True
        replay_tile = tile.clone().requires_grad_()
        replay = streaming.reduce_tile_for_backward(
            replay_tile, valid, streaming.extra_state_for_backward()
        )
        replay.backward(upstream)
        replay_grad[..., y0:y1, x0:x1] += replay_tile.grad
    assert torch.allclose(replay_grad, expected_x_grad, atol=1e-10)
    if learnable:
        assert torch.allclose(streaming.raw_tau.grad, expected_tau_grad, atol=1e-9)


def test_conversion_and_invalid_configuration():
    reducer = SigmoidAttentionPoolingReducer(
        0.4, True, True, torch.float64, True, "bilinear"
    )
    streaming = reducer.to_streaming()
    assert isinstance(streaming, StreamingSigmoidAttentionPoolingReducer)
    assert isinstance(streaming.raw_tau, torch.nn.Parameter)
    assert streaming.accumulator_dtype == torch.float64 and streaming.mask_resize
    restored = streaming.to_reducer()
    assert isinstance(restored.raw_tau, torch.nn.Parameter)
    assert torch.equal(restored.raw_tau, reducer.raw_tau)
    with pytest.raises(ValueError):
        SigmoidAttentionPoolingReducer(0)
    with pytest.raises(ValueError):
        reducer(torch.randn(2, 3, 4))
    with pytest.raises(ValueError):
        reducer(torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2))
