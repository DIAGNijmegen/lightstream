from types import SimpleNamespace
import pytest
import torch
from lightstream.core.reducer import (
    SigmoidAttentionPoolingReducer,
    StreamingSigmoidAttentionPoolingReducer,
)


def _shifted_softmax_reference(x, tau, mask=None):
    q = torch.sigmoid(x) / tau
    valid = None if mask is None else mask.to(device=x.device, dtype=torch.bool)
    if valid is not None:
        q = q.masked_fill(~valid, torch.finfo(x.dtype).min)
    m = q.amax(dim=(-2, -1), keepdim=True)
    exp_shifted = torch.exp(q - m)
    if valid is not None:
        exp_shifted = torch.where(valid, exp_shifted, torch.zeros_like(exp_shifted))
    z = exp_shifted.sum(dim=(-2, -1), keepdim=True, dtype=x.dtype)
    weights = exp_shifted / z.clamp_min(torch.finfo(x.dtype).tiny)
    return (weights * x).sum(dim=(-2, -1), keepdim=True, dtype=x.dtype)


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


def test_offline_matches_explicit_shifted_softmax_reference():
    torch.manual_seed(29)
    x = torch.randn(2, 3, 7, 11, dtype=torch.float64)
    mask = torch.rand(2, 1, 7, 11) > 0.25
    mask[1] = False
    reducer = SigmoidAttentionPoolingReducer(0.37, accumulator_dtype=torch.float64)

    actual = reducer(x, mask=mask)
    expected = _shifted_softmax_reference(x, reducer.current_tau, mask)
    expected = torch.where(
        mask.flatten(2).any(-1, keepdim=True).unsqueeze(-1),
        expected,
        torch.zeros_like(expected),
    )

    # The fused softmax and its explicit shifted decomposition are numerically
    # equivalent, but are not required to choose bitwise-identical reductions.
    assert torch.allclose(actual, expected, rtol=1e-14, atol=1e-14)


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
    streamed = streaming.finish_stream()
    shifted_reference = offline(x.detach())
    q = torch.sigmoid(x.detach()) / offline.current_tau
    fused_reference = (torch.softmax(q.flatten(2), -1).view_as(q) * x.detach()).sum(
        dim=(-2, -1), keepdim=True, dtype=torch.float64
    )
    shifted_error = (streamed - shifted_reference).abs().max().item()
    fused_error = (streamed - fused_reference).abs().max().item()
    assert (
        shifted_error <= 2e-16
    ), f"explicit shifted-softmax reference error: {shifted_error}"
    assert fused_error <= 3e-16, f"fused torch.softmax reference error: {fused_error}"
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


@pytest.mark.parametrize("stopgrad", [False, True])
@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_backward_replay_input_gradient_matches_direct_offline_autograd(
    stopgrad, learnable, use_mask, reverse_order
):
    """Exercise overlap ownership and replay order at float64 precision."""
    torch.manual_seed(319)
    shape = (2, 3, 5, 7)
    source = torch.randn(*shape, dtype=torch.float64)
    upstream = torch.randn(2, 3, 1, 1, dtype=torch.float64)
    spatial_mask = torch.tensor(
        [
            [1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 0, 1, 1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    mask = spatial_mask if use_mask else torch.ones(5, 7, dtype=torch.bool)

    # Deliberately overlapping tiles; reversing them changes which tile owns
    # every overlap while preserving the global set of valid pixels.
    boxes = [
        (0, 3, 0, 4),
        (0, 3, 3, 7),
        (2, 5, 0, 4),
        (2, 5, 3, 7),
    ]
    if reverse_order:
        boxes.reverse()

    streaming = StreamingSigmoidAttentionPoolingReducer(
        0.73,
        learnable_temperature=learnable,
        stopgrad_attention=stopgrad,
        accumulator_dtype=torch.float64,
    )
    streaming.start_stream(5, 7, 2, 3, source.device, source.dtype)
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    for y0, y1, x0, x1 in boxes:
        tile = source[..., y0:y1, x0:x1]
        streaming.accumulate_stream_tile(
            tile,
            y0,
            x0,
            sides,
            (y0, y1, x0, x1),
            user_mask=mask[y0:y1, x0:x1],
        )
    streaming.finish_stream()

    direct_x = source.clone().requires_grad_()
    direct_scores = torch.sigmoid(direct_x)
    if stopgrad:
        direct_scores = direct_scores.detach()
    direct_q = direct_scores / streaming.current_tau
    direct_q = direct_q.masked_fill(~mask, -torch.inf)
    direct_weights = torch.softmax(direct_q.flatten(2), dim=-1).view_as(direct_x)
    direct_output = (direct_weights * direct_x).sum((-2, -1), keepdim=True)
    direct_output.backward(upstream)
    expected_input_grad = direct_x.grad.detach().clone()
    expected_tau_grad = (
        streaming.raw_tau.grad.detach().clone() if learnable else None
    )
    if learnable:
        streaming.raw_tau.grad = None

    replay_input_grad = torch.zeros_like(source)
    seen = torch.zeros(5, 7, dtype=torch.bool)
    context = streaming.extra_state_for_backward()
    for y0, y1, x0, x1 in boxes:
        new = ~seen[y0:y1, x0:x1]
        valid = new & mask[y0:y1, x0:x1]
        seen[y0:y1, x0:x1] = True
        replay_tile = source[..., y0:y1, x0:x1].clone().requires_grad_()
        replay = streaming.reduce_tile_for_backward(replay_tile, valid, context)
        replay.backward(upstream)
        replay_input_grad[..., y0:y1, x0:x1] += replay_tile.grad

    assert torch.allclose(
        replay_input_grad, expected_input_grad, rtol=2e-13, atol=2e-13
    )
    if learnable:
        assert torch.allclose(
            streaming.raw_tau.grad, expected_tau_grad, rtol=2e-13, atol=2e-13
        )


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("learnable", [False, True])
def test_temperature_conversion_preserves_exact_raw_and_effective_values(
    dtype, learnable
):
    """Reproduce conversion followed by the final model dtype/device move."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offline = SigmoidAttentionPoolingReducer(
        tau_init=0.37,
        learnable_temperature=learnable,
        tau_min=1e-6,
    ).to(device=device, dtype=dtype)

    before = {
        "raw_tau": offline.raw_tau.detach().clone(),
        "current_tau": offline.current_tau.detach().clone(),
    }
    streaming = offline.to_streaming()
    # StreamingWSS performs conversion while it is constructed and applies
    # this final model-wide move afterwards.
    streaming.to(device=device, dtype=dtype)
    during = {
        "raw_tau": streaming.raw_tau.detach().clone(),
        "current_tau": streaming.current_tau.detach().clone(),
    }
    restored = streaming.to_reducer()
    after = {
        "raw_tau": restored.raw_tau.detach().clone(),
        "current_tau": restored.current_tau.detach().clone(),
    }

    for name in ("raw_tau", "current_tau"):
        reference = before[name]
        for stage, actual in (("streaming", during[name]), ("restored", after[name])):
            difference = torch.abs(actual - reference)
            diagnostic = (
                f"{name} at {stage}: dtype={actual.dtype}, device={actual.device}, "
                f"scalar={actual.item()!r}, reference={reference.item()!r}, "
                f"absolute_difference={difference.item()!r}, "
                f"torch.equal={torch.equal(actual, reference)}"
            )
            assert actual.dtype == reference.dtype, diagnostic
            assert actual.device == reference.device, diagnostic
            assert difference.item() == 0.0, diagnostic
            assert torch.equal(actual, reference), diagnostic

    assert isinstance(streaming.raw_tau, torch.nn.Parameter) == learnable
    assert isinstance(restored.raw_tau, torch.nn.Parameter) == learnable
