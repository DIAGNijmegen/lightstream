"""Contract and streaming-equivalence checks for nGWP and size-focal reducers."""

from types import SimpleNamespace

import pytest
import torch

from lightstream.core.reducer import NGWPReducer, SizeFocalReducer, StreamingNGWPReducer, StreamingSizeFocalReducer


def _stream(reducer, inputs, mask=None):
    x = inputs[0]
    stream = reducer.to_streaming()
    stream.start_stream(x.shape[-2], x.shape[-1], x.shape[0], x.shape[1], x.device, x.dtype)
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    for y0, y1 in ((0, 2), (2, x.shape[-2])):
        for x0, x1 in ((0, 2), (2, x.shape[-1])):
            payload = tuple(v[..., y0:y1, x0:x1] for v in inputs)
            stream.accumulate_stream_tile(payload if len(payload) > 1 else payload[0], y0, x0, sides, (y0, y1, x0, x1), None if mask is None else mask[y0:y1, x0:x1])
    return stream.finish_stream()


def test_offline_formulas_masks_empty_masks_and_resize():
    scores = torch.tensor([[[[1., 3.], [5., 7.]]]])
    activation = torch.tensor([[[[.2, .4], [.6, .8]]]])
    mask = torch.tensor([[True, False], [True, False]])
    ngwp = NGWPReducer(eps=.5)
    expected = (scores * activation * mask).sum((-2, -1), keepdim=True) / (.5 + (activation * mask).sum((-2, -1), keepdim=True))
    assert torch.allclose(ngwp(scores, activation, mask=mask), expected)
    assert torch.equal(ngwp(scores, activation, mask=torch.zeros_like(mask)), torch.zeros(1, 1, 1, 1))

    m = torch.tensor([[[[.2, .4], [.6, .8]]]])
    focal = SizeFocalReducer(p=2, lambda_=1.)
    mean = (m * mask).sum() / mask.sum()
    assert torch.allclose(focal(m, mask=mask), (1 - mean).pow(2) * torch.log(1 + mean))
    assert torch.equal(focal(m, mask=torch.zeros_like(mask)), torch.zeros(1, 1, 1, 1))
    small = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError): focal(torch.ones(1, 1, 4, 4), mask=small)
    assert SizeFocalReducer(lambda_=1, mask_resize=True)(torch.ones(1, 1, 4, 4), mask=small).shape == (1, 1, 1, 1)
    assert NGWPReducer(mask_resize=True)(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), mask=small).shape == (1, 1, 1, 1)


def test_validation_conversion_tiled_forward_and_backward_equivalence():
    with pytest.raises(ValueError): NGWPReducer()(torch.ones(1, 1, 2, 2))
    with pytest.raises(ValueError): NGWPReducer()(torch.ones(1, 1, 2, 2), torch.ones(1, 2, 2, 2))
    with pytest.raises(ValueError): SizeFocalReducer(p=float("nan"))
    with pytest.raises(ValueError): SizeFocalReducer(lambda_=0)
    assert isinstance(NGWPReducer().to_streaming(), StreamingNGWPReducer)
    assert isinstance(SizeFocalReducer().to_streaming(), StreamingSizeFocalReducer)

    torch.manual_seed(9)
    mask = torch.rand(4, 5) > .3
    cases = [(NGWPReducer(eps=.1), (torch.rand(2, 3, 4, 5, dtype=torch.float64), torch.rand(2, 3, 4, 5, dtype=torch.float64))),
             (SizeFocalReducer(p=2, lambda_=1), (torch.rand(2, 3, 4, 5, dtype=torch.float64),))]
    for reducer, values in cases:
        assert torch.allclose(_stream(reducer, values, mask), reducer(*values, mask=mask), atol=1e-10)
        inputs = [v.detach().requires_grad_() for v in values]
        upstream = torch.randn(2, 3, 1, 1, dtype=torch.float64)
        reducer(*inputs, mask=mask).backward(upstream)
        expected = [v.grad.clone() for v in inputs]
        stream = reducer.to_streaming(); stream.accumulate_valid_tile(tuple(v.detach() for v in inputs) if len(inputs) > 1 else inputs[0].detach(), mask)
        replay = [v.detach().requires_grad_() for v in inputs]
        tile = tuple(replay) if len(replay) > 1 else replay[0]
        stream.reduce_tile_for_backward(tile, mask, stream.extra_state_for_backward()).backward(upstream)
        for actual, wanted in zip(replay, expected): assert torch.allclose(actual.grad, wanted, atol=1e-10)
