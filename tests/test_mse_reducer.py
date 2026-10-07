"""Full-frame and tiled checks for spatial consistency MSE."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lightstream.core.reducer import MSEReducer, StreamingMSEReducer
from lightstream.core.scnn.scnn import StreamingCNN


def _reference(x, mask, use_softmax):
    valid = mask[None, None]
    count = valid.sum().clamp_min(1)
    if use_softmax:
        masked = x.masked_fill(~valid, -torch.inf)
        probabilities = torch.softmax(masked.flatten(2), dim=-1).view_as(x)
        values = torch.where(valid, probabilities, torch.zeros_like(probabilities))
    else:
        values = x
    mean = torch.where(valid, values, torch.zeros_like(values)).sum(
        (-2, -1), keepdim=True
    ) / count
    centered = torch.where(valid, values - mean, torch.zeros_like(values))
    return centered.square().sum((-2, -1), keepdim=True) / count


@pytest.mark.parametrize("use_softmax", [False, True])
def test_formula_mask_and_gradients(use_softmax):
    x = torch.tensor(
        [[[[2.0, -1.0, 5.0], [4.0, 8.0, 3.0]],
          [[-3.0, 2.0, 7.0], [1.0, 4.0, -2.0]]]],
        dtype=torch.float64, requires_grad=True,
    )
    mask = torch.tensor([[True, False, True], [True, True, False]])
    upstream = torch.tensor([[[[0.8]], [[-1.3]]]], dtype=torch.float64)
    reducer = MSEReducer(use_softmax=use_softmax, accumulator_dtype=torch.float64)
    expected_input = x.detach().clone().requires_grad_()
    expected = _reference(expected_input, mask, use_softmax)
    actual = reducer(x, mask=mask)
    assert actual.shape == (1, 2, 1, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.autograd.backward(actual, upstream)
    torch.autograd.backward(expected, upstream)
    torch.testing.assert_close(x.grad, expected_input.grad, rtol=1e-11, atol=1e-12)
    assert torch.count_nonzero(x.grad[..., ~mask]) == 0


def test_softmax_option_matches_pre_normalized_input():
    x = torch.tensor([[[[0.0, 2.0], [-3.0, 4.0]]]], dtype=torch.float64)
    probabilities = torch.softmax(x.flatten(2), dim=-1).view_as(x)
    torch.testing.assert_close(
        MSEReducer(use_softmax=True)(x),
        MSEReducer()(probabilities),
    )
    uniform = torch.zeros_like(x)
    torch.testing.assert_close(
        MSEReducer(use_softmax=True)(uniform),
        torch.zeros((1, 1, 1, 1), dtype=x.dtype),
    )


def test_extreme_logits_and_low_precision_output():
    logits = torch.tensor([[[[1000.0, -1000.0], [999.0, -999.0]]]])
    expected = _reference(logits.double(), torch.ones(2, 2, dtype=torch.bool), True)
    actual = MSEReducer(use_softmax=True, accumulator_dtype=torch.float64)(logits)
    assert actual.dtype == logits.dtype and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-7)
    half = logits.half()
    assert MSEReducer(use_softmax=True)(half).dtype == torch.float16


@pytest.mark.parametrize("use_softmax", [False, True])
def test_empty_mask_resize_validation_and_conversion(use_softmax):
    x = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    empty = torch.zeros(3, 4, dtype=torch.bool)
    reducer = MSEReducer(use_softmax=use_softmax)
    result = reducer(x, mask=empty)
    torch.testing.assert_close(result, torch.zeros_like(result))
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    with pytest.raises(ValueError):
        reducer(x, mask=torch.ones(1, 1, dtype=torch.bool))
    assert MSEReducer(mask_resize=True)(x, mask=torch.ones(1, 1, dtype=torch.bool)).shape == (1, 2, 1, 1)
    with pytest.raises(ValueError):
        reducer(x, x)
    with pytest.raises(ValueError):
        reducer(torch.ones(2, 3))
    with pytest.raises(TypeError):
        reducer(torch.ones(1, 1, 2, 2, dtype=torch.int64))
    streamed = reducer.to_streaming()
    assert isinstance(streamed, StreamingMSEReducer)
    assert streamed.use_softmax is use_softmax
    assert streamed.to_reducer().use_softmax is use_softmax


def test_independent_mse_heads_have_distinct_passthrough_identities():
    x = torch.randn(1, 2, 3, 4, requires_grad=True)
    first = MSEReducer().to_streaming()
    second = MSEReducer(use_softmax=True).to_streaming()
    first_output, second_output = first(x), second(x)
    assert first_output is not second_output
    assert first_output.data_ptr() == second_output.data_ptr() == x.data_ptr()
    assert first_output.requires_grad and second_output.requires_grad


@pytest.mark.parametrize("use_softmax", [False, True])
def test_tiled_forward_and_replay_match_full_frame(use_softmax):
    torch.manual_seed(173)
    x = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
    mask = torch.rand(5, 7) > 0.3
    reducer = MSEReducer(use_softmax=use_softmax, accumulator_dtype=torch.float64)
    upstream = torch.randn(2, 3, 1, 1, dtype=torch.float64)
    expected = reducer(x, mask=mask)
    torch.autograd.backward(expected, upstream)

    stream = reducer.to_streaming()
    stream.start_stream(5, 7, 2, 3, x.device, x.dtype)
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    for y0, y1 in ((0, 2), (2, 5)):
        for x0, x1 in ((0, 3), (3, 7)):
            tile = x.detach()[..., y0:y1, x0:x1]
            valid = mask[y0:y1, x0:x1]
            stream.accumulate_stream_tile(tile, y0, x0, sides, (y0, y1, x0, x1), valid)
    torch.testing.assert_close(stream.finish_stream(), expected, rtol=1e-11, atol=1e-12)

    replay_x = x.detach().clone().requires_grad_()
    context = stream.extra_state_for_backward()
    replay = 0
    for y0, y1 in ((0, 2), (2, 5)):
        for x0, x1 in ((0, 3), (3, 7)):
            replay = replay + stream.reduce_tile_for_backward(
                replay_x[..., y0:y1, x0:x1], mask[y0:y1, x0:x1], context
            )
    torch.autograd.backward(replay, upstream)
    torch.testing.assert_close(replay_x.grad, x.grad, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("use_softmax", [False, True])
def test_fully_masked_stream_returns_zero(use_softmax):
    x = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    stream = MSEReducer(use_softmax=use_softmax).to_streaming()
    stream.start_stream(3, 4, 1, 2, x.device, x.dtype)
    sides = SimpleNamespace(top=True, left=True, right=True, bottom=True)
    stream.accumulate_stream_tile(x, 0, 0, sides, (0, 3, 0, 4), torch.zeros(3, 4, dtype=torch.bool))
    torch.testing.assert_close(stream.finish_stream(), torch.zeros(1, 2, 1, 1, dtype=x.dtype))
    replay_x = x.clone().requires_grad_()
    replay = stream.reduce_tile_for_backward(
        replay_x, torch.zeros(3, 4, dtype=torch.bool), stream.extra_state_for_backward()
    )
    replay.sum().backward()
    torch.testing.assert_close(replay_x.grad, torch.zeros_like(replay_x))


class _MSEHead(nn.Module):
    def __init__(self, use_softmax):
        super().__init__()
        self.producer = nn.Conv2d(3, 2, kernel_size=1, dtype=torch.float64)
        self.reducer = MSEReducer(use_softmax=use_softmax, accumulator_dtype=torch.float64)

    def forward(self, x):
        return self.reducer(self.producer(x))


@pytest.mark.parametrize("use_softmax", [False, True])
def test_scnn_shifted_tiles_forward_and_backward(use_softmax):
    torch.manual_seed(179)
    x = torch.randn(1, 3, 5, 7, dtype=torch.float64)
    upstream = torch.randn(1, 2, 1, 1, dtype=torch.float64)
    reference = _MSEHead(use_softmax)
    streamed_model = copy.deepcopy(reference)
    ref_x = x.clone().requires_grad_()
    torch.autograd.backward(reference(ref_x), upstream)
    streamed = StreamingCNN(
        streamed_model, tile_shape=(1, 3, 4, 4), deterministic=True,
        saliency=False, copy_to_gpu=False, statistics_on_cpu=False,
        normalize_on_gpu=False,
    )
    stream_x = x.clone().requires_grad_()
    torch.testing.assert_close(streamed(stream_x), reference(x), rtol=1e-10, atol=1e-12)
    streamed.backward(stream_x, upstream)
    torch.testing.assert_close(stream_x.grad, ref_x.grad, rtol=1e-9, atol=1e-11)
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(
            dict(streamed_model.named_parameters())[name].grad,
            parameter.grad, rtol=1e-9, atol=1e-11,
        )
