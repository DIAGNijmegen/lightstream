from types import SimpleNamespace

import pytest
import torch

from lightstream.core.reducer import (
    NormalizedSigmoidAttentionReducer,
    StreamingNormalizedSigmoidAttentionReducer,
)


def _reference(values, logits, mask=None):
    if logits.ndim == 3:
        logits = logits[:, None]
    elif logits.shape[1] == values.shape[1]:
        logits = logits.mean(1, keepdim=True)
    scores = torch.sigmoid(logits)
    if mask is not None:
        scores = scores * mask[:, None]
    denominator = scores.sum((-2, -1), keepdim=True)
    return torch.where(
        denominator > 0,
        (scores * values).sum((-2, -1), keepdim=True) / denominator.clamp_min(torch.finfo(values.dtype).tiny),
        torch.zeros_like(values[..., :1, :1]),
    )


@pytest.mark.parametrize("logit_layout", ["nhw", "n1hw", "nchw"])
def test_reference_formula_channel_contract_and_signed_values(logit_layout):
    values = torch.tensor([[[[-5.0, 2.0], [7.0, -3.0]], [[1.0, -4.0], [2.0, 8.0]]]], dtype=torch.float64)
    base = torch.tensor([[[[-2.0, 0.5], [1.0, 3.0]]]], dtype=torch.float64)
    logits = {"nhw": base[:, 0], "n1hw": base, "nchw": base.expand_as(values)}[logit_layout]
    actual = NormalizedSigmoidAttentionReducer()(values, logits)
    assert torch.allclose(actual, _reference(values, logits))
    softmax_of_sigmoid = (torch.softmax(torch.sigmoid(base).flatten(2), -1).view_as(base) * values).sum((-2, -1), keepdim=True)
    assert not torch.allclose(actual, softmax_of_sigmoid)


def test_presigmoided_values_are_not_transformed_and_masks_are_safe():
    values = torch.sigmoid(torch.tensor([[[[-3.0, 0.0], [2.0, 4.0]]], [[[1.0, -2.0], [3.0, 0.5]]]]))
    logits = torch.randn(2, 1, 2, 2)
    mask = torch.tensor([[[True, False], [True, False]], [[False, False], [False, False]]])
    result = NormalizedSigmoidAttentionReducer()(values, logits, mask=mask)
    assert torch.allclose(result, _reference(values, logits, mask))
    assert torch.equal(result[1], torch.zeros_like(result[1]))


def test_accumulator_dtype_and_offline_gradients():
    values = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    reducer = NormalizedSigmoidAttentionReducer(accumulator_dtype=torch.float64)
    output = reducer(values, logits)
    expected = _reference(values, logits)
    grads = torch.autograd.grad(output.sum(), (values, logits))
    expected_grads = torch.autograd.grad(expected.sum(), (values, logits))
    assert output.dtype == torch.float64
    assert torch.allclose(output, expected)
    assert all(torch.allclose(a, b) for a, b in zip(grads, expected_grads))

    low = torch.randn(1, 2, 3, 4, dtype=torch.float16)
    assert reducer.to(dtype=torch.float32)(low, logits.detach().float()).dtype == torch.float16


def test_streaming_forward_backward_and_conversions_across_tiles():
    torch.manual_seed(31)
    values = torch.randn(1, 2, 4, 6, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(1, 1, 4, 6, dtype=torch.float64, requires_grad=True)
    reducer = NormalizedSigmoidAttentionReducer(accumulator_dtype=torch.float64)
    upstream = torch.randn(1, 2, 1, 1, dtype=torch.float64)
    torch.autograd.backward(reducer(values, logits), upstream)
    expected_grads = values.grad.clone(), logits.grad.clone()

    streaming = reducer.to_streaming()
    assert isinstance(streaming, StreamingNormalizedSigmoidAttentionReducer)
    streaming.start_stream(4, 6, 1, 2, values.device, values.dtype)
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    replay_tiles = []
    for x0, x1 in ((0, 2), (2, 6)):
        payload = (values.detach()[..., x0:x1], logits.detach()[..., x0:x1])
        streaming.accumulate_stream_tile(payload, 0, x0, sides, (0, 4, x0, x1))
        replay_tiles.append((x0, x1))
    assert torch.allclose(streaming.finish_stream(), reducer(values.detach(), logits.detach()))

    replay_values = values.detach().clone().requires_grad_(True)
    replay_logits = logits.detach().clone().requires_grad_(True)
    context = streaming.extra_state_for_backward()
    for x0, x1 in replay_tiles:
        tile_result = streaming.reduce_tile_for_backward(
            (replay_values[..., x0:x1], replay_logits[..., x0:x1]),
            torch.ones(4, x1 - x0, dtype=torch.bool), context,
        )
        torch.autograd.backward(tile_result, upstream)
    assert torch.allclose(replay_values.grad, expected_grads[0])
    assert torch.allclose(replay_logits.grad, expected_grads[1])
    assert isinstance(streaming.to_reducer(), NormalizedSigmoidAttentionReducer)
