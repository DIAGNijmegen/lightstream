from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


from lightstream.core.reducer import AttentionGeMReducer, FusedAttentionGeMReducer


def _attention_formula(x, logits, *, r, eps, uniform_attention_eps, mask=None):
    x_pow = x.clamp_min(eps).pow(torch.as_tensor(r, dtype=x.dtype, device=x.device))
    if mask is None:
        valid = torch.ones(x.shape[-2:], dtype=torch.bool, device=x.device)
    else:
        valid = mask.to(device=x.device, dtype=torch.bool)
    n_valid = int(valid.sum().item())
    assert n_valid == (x.shape[-2] * x.shape[-1] if mask is None else int(mask.sum().item()))

    logits_flat = logits.reshape(x.shape[0], 1, -1)
    valid_flat = valid.flatten()
    att = torch.zeros_like(logits_flat)
    att[..., valid_flat] = torch.softmax(logits_flat[..., valid_flat], dim=-1)
    mixed = (1.0 - uniform_attention_eps) * att
    mixed[..., valid_flat] = mixed[..., valid_flat] + uniform_attention_eps / n_valid
    return (mixed.reshape_as(logits) * x_pow).sum(dim=(-2, -1), keepdim=True).clamp_min(eps).pow(1.0 / r)


def _fused_formula(y1, y2, y3, logits, *, r, eps, value_weights, attention_weights, uniform_attention_eps, mask=None):
    fused_y = value_weights[0] * y1 + value_weights[1] * y2 + value_weights[2] * y3
    x_pow = fused_y.clamp_min(eps).pow(torch.as_tensor(r, dtype=fused_y.dtype, device=fused_y.device))
    if mask is None:
        valid = torch.ones(fused_y.shape[-2:], dtype=torch.bool, device=fused_y.device)
    else:
        valid = mask.to(device=fused_y.device, dtype=torch.bool)
    n_valid = int(valid.sum().item())
    assert n_valid == (fused_y.shape[-2] * fused_y.shape[-1] if mask is None else int(mask.sum().item()))

    valid_flat = valid.flatten()
    fused_attention = torch.zeros((fused_y.shape[0], 1, fused_y.shape[-2] * fused_y.shape[-1]), dtype=fused_y.dtype, device=fused_y.device)
    for branch_weight, branch_logits in zip(attention_weights, logits):
        branch_flat = branch_logits.reshape(fused_y.shape[0], 1, -1)
        branch_attention = torch.zeros_like(branch_flat)
        branch_attention[..., valid_flat] = torch.softmax(branch_flat[..., valid_flat], dim=-1)
        fused_attention = fused_attention + branch_weight * branch_attention

    mixed = (1.0 - uniform_attention_eps) * fused_attention
    mixed[..., valid_flat] = mixed[..., valid_flat] + uniform_attention_eps / n_valid
    return (mixed.reshape(fused_y.shape[0], 1, *fused_y.shape[-2:]) * x_pow).sum(dim=(-2, -1), keepdim=True).clamp_min(eps).pow(1.0 / r)


def _stream_attention(reducer, x, logits, mask):
    streaming = reducer.to_streaming()
    streaming.start_stream(x.shape[-2], x.shape[-1], x.shape[0], x.shape[1], x.device, x.dtype)
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    h_mid = x.shape[-2] // 2
    w_mid = x.shape[-1] // 2
    for y0, y1 in ((0, h_mid), (h_mid, x.shape[-2])):
        for x0, x1 in ((0, w_mid), (w_mid, x.shape[-1])):
            tile = (x[..., y0:y1, x0:x1], logits[..., y0:y1, x0:x1])
            user_mask = None if mask is None else mask[y0:y1, x0:x1]
            streaming.accumulate_stream_tile(tile, y0, x0, sides, (y0, y1, x0, x1), user_mask=user_mask)
    return streaming.finish_stream()


def _stream_fused(reducer, inputs, mask):
    y1, y2, y3, *logits = inputs
    streaming = reducer.to_streaming()
    streaming.start_stream(y1.shape[-2], y1.shape[-1], y1.shape[0], y1.shape[1], y1.device, y1.dtype)
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    h_mid = y1.shape[-2] // 2
    w_mid = y1.shape[-1] // 2
    for y0, y1_idx in ((0, h_mid), (h_mid, y1.shape[-2])):
        for x0, x1_idx in ((0, w_mid), (w_mid, y1.shape[-1])):
            sliced = [tensor[..., y0:y1_idx, x0:x1_idx] for tensor in inputs]
            payload = streaming(*sliced)
            user_mask = None if mask is None else mask[y0:y1_idx, x0:x1_idx]
            streaming.accumulate_stream_tile(payload, y0, x0, sides, (y0, y1_idx, x0, x1_idx), user_mask=user_mask)
    return streaming.finish_stream()


@pytest.mark.parametrize("mask", [None, torch.tensor([[True, False, True, True], [True, True, False, True], [False, True, True, True]])])
def test_attention_gem_uniform_mix_matches_direct_formula_masked_and_unmasked(mask):
    torch.manual_seed(201)
    x = torch.rand(2, 3, 3, 4, dtype=torch.float64) + 0.1
    logits = torch.randn(2, 1, 3, 4, dtype=torch.float64)
    reducer = AttentionGeMReducer(r_init=1.7, eps=1e-9, uniform_attention_eps=0.35)

    expected = _attention_formula(x, logits, r=reducer.current_r, eps=1e-9, uniform_attention_eps=0.35, mask=mask)

    assert torch.allclose(reducer(x, logits, mask=mask), expected, atol=1e-8, rtol=1e-8)
    assert torch.allclose(_stream_attention(reducer, x, logits, mask), expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("mask", [None, torch.tensor([[True, False, True, True], [True, True, False, True], [False, True, True, True]])])
def test_fused_attention_gem_uniform_mix_matches_direct_formula_after_branch_fusion_masked_and_unmasked(mask):
    torch.manual_seed(203)
    y1 = torch.rand(2, 3, 3, 4, dtype=torch.float64) + 0.1
    y2 = torch.rand(2, 3, 3, 4, dtype=torch.float64) + 0.1
    y3 = torch.rand(2, 3, 3, 4, dtype=torch.float64) + 0.1
    logits = tuple(torch.randn(2, 1, 3, 4, dtype=torch.float64) for _ in range(3))
    value_weights = (0.2, 0.5, 0.3)
    attention_weights = (0.25, 0.35, 0.4)
    reducer = FusedAttentionGeMReducer(
        r_init=1.8,
        eps=1e-9,
        value_weights=value_weights,
        attention_weights=attention_weights,
        uniform_attention_eps=0.3,
    )

    expected = _fused_formula(
        y1,
        y2,
        y3,
        logits,
        r=reducer.current_r,
        eps=1e-9,
        value_weights=reducer.value_weights,
        attention_weights=reducer.attention_weights,
        uniform_attention_eps=0.3,
        mask=mask,
    )

    assert torch.allclose(reducer(y1, y2, y3, *logits, mask=mask), expected, atol=1e-8, rtol=1e-8)
    assert torch.allclose(_stream_fused(reducer, (y1, y2, y3, *logits), mask), expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("mask", [None, torch.tensor([[True, True, False, True], [False, True, True, True], [True, False, True, True]])])
def test_attention_gem_streaming_backward_replay_matches_nonstreaming_value_and_logits(mask):
    torch.manual_seed(205)
    x = (torch.rand(2, 3, 3, 4, dtype=torch.float64) + 0.1).requires_grad_(True)
    logits = torch.randn(2, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    reducer = AttentionGeMReducer(r_init=1.6, eps=1e-9, uniform_attention_eps=0.2)
    upstream = torch.randn(2, 3, 1, 1, dtype=torch.float64)

    torch.autograd.backward(reducer(x, logits, mask=mask), upstream)
    expected_x_grad = x.grad.detach().clone()
    expected_logits_grad = logits.grad.detach().clone()

    state_streaming = reducer.to_streaming()
    valid_mask = torch.ones(x.shape[-2:], dtype=torch.bool) if mask is None else mask
    state_streaming.accumulate_valid_tile((x.detach(), logits.detach()), valid_mask=valid_mask)

    replay_x = x.detach().clone().requires_grad_(True)
    replay_logits = logits.detach().clone().requires_grad_(True)
    replay = state_streaming.reduce_tile_for_backward(
        (replay_x, replay_logits),
        valid_mask=valid_mask,
        global_context=state_streaming.extra_state_for_backward(),
    )
    torch.autograd.backward(replay, upstream)

    assert torch.allclose(replay_x.grad, expected_x_grad, atol=1e-9, rtol=1e-8)
    assert torch.allclose(replay_logits.grad, expected_logits_grad, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("mask", [None, torch.tensor([[True, True, False, True], [False, True, True, True], [True, False, True, True]])])
def test_fused_attention_gem_streaming_backward_replay_matches_nonstreaming_values_and_logits(mask):
    torch.manual_seed(207)
    value_weights = (0.2, 0.5, 0.3)
    attention_weights = (0.25, 0.35, 0.4)
    reducer = FusedAttentionGeMReducer(
        r_init=1.6,
        eps=1e-9,
        value_weights=value_weights,
        attention_weights=attention_weights,
        uniform_attention_eps=0.2,
    )
    values = [(torch.rand(2, 3, 3, 4, dtype=torch.float64) + 0.1).requires_grad_(True) for _ in range(3)]
    logits = [torch.randn(2, 1, 3, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    upstream = torch.randn(2, 3, 1, 1, dtype=torch.float64)

    torch.autograd.backward(reducer(*values, *logits, mask=mask), upstream)
    expected_value_grads = [value.grad.detach().clone() for value in values]
    expected_logit_grads = [logit.grad.detach().clone() for logit in logits]

    state_streaming = reducer.to_streaming()
    valid_mask = torch.ones(values[0].shape[-2:], dtype=torch.bool) if mask is None else mask
    with torch.no_grad():
        state_payload = state_streaming(*(tensor.detach() for tensor in (*values, *logits)))
    state_streaming.accumulate_valid_tile(state_payload, valid_mask=valid_mask)

    replay_values = [value.detach().clone().requires_grad_(True) for value in values]
    replay_logits = [logit.detach().clone().requires_grad_(True) for logit in logits]
    replay_payload = state_streaming(*replay_values, *replay_logits)
    replay = state_streaming.reduce_tile_for_backward(
        replay_payload,
        valid_mask=valid_mask,
        global_context=state_streaming.extra_state_for_backward(),
    )
    torch.autograd.backward(replay, upstream)

    for actual, expected in zip(replay_values, expected_value_grads):
        assert torch.allclose(actual.grad, expected, atol=1e-9, rtol=1e-8)
    for actual, expected in zip(replay_logits, expected_logit_grads):
        assert torch.allclose(actual.grad, expected, atol=1e-9, rtol=1e-8)
