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


def _sigmoid_attention_state_reference(x, tau, accepted, accumulator_dtype):
    """Recompute the streaming state from the full frame, not from tiles."""
    values = x.to(accumulator_dtype)
    q = torch.sigmoid(values) / tau.to(device=x.device, dtype=accumulator_dtype)
    valid = accepted.to(device=x.device)[None, None]
    q = torch.where(valid, q, torch.full_like(q, torch.finfo(q.dtype).min))
    m = q.amax(dim=(-2, -1), keepdim=True)
    shifted = torch.where(valid, torch.exp(q - m), torch.zeros_like(q))
    zhat = shifted.sum(dim=(-2, -1), keepdim=True, dtype=accumulator_dtype)
    shat = (shifted * values).sum(dim=(-2, -1), keepdim=True, dtype=accumulator_dtype)
    return m, zhat, shat


def _state_error(actual, expected):
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
    return absolute.max().item(), relative.max().item()


def _state_matches(actual, expected, *, atol, rtol):
    error = (actual - expected).abs()
    return bool(torch.all(error <= atol + rtol * expected.abs()))


def _max_absolute_error(actual, expected):
    return (actual - expected).abs().max().item()


def _precision_limit(reference):
    """A small, scale-aware envelope around float64 machine precision."""
    scale = max(1.0, reference.detach().abs().max().item())
    return 8 * torch.finfo(reference.dtype).eps * scale


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


@pytest.mark.parametrize("reverse_order", [False, True], ids=["forward", "reversed"])
def test_streaming_state_matches_full_frame_reference_after_every_tile(reverse_order):
    """Diagnose the first merge that departs from the defining full-frame formula."""
    accumulator_dtype = torch.float64
    source = -torch.tensor(
        [
            [
                [
                    [8.0, 1.0, 4.0, 9.0, 2.0, 7.0],
                    [3.0, 6.0, 2.0, 5.0, 8.0, 1.0],
                    [9.0, 4.0, 7.0, 3.0, 6.0, 2.0],
                    [2.0, 8.0, 1.0, 6.0, 4.0, 9.0],
                ],
                [
                    [2.0, 7.0, 5.0, 1.0, 8.0, 4.0],
                    [6.0, 3.0, 9.0, 2.0, 5.0, 7.0],
                    [1.0, 8.0, 4.0, 6.0, 2.0, 9.0],
                    [7.0, 2.0, 6.0, 4.0, 9.0, 3.0],
                ],
                [
                    [5.0, 9.0, 2.0, 7.0, 3.0, 6.0],
                    [8.0, 1.0, 4.0, 9.0, 6.0, 2.0],
                    [3.0, 7.0, 5.0, 1.0, 8.0, 4.0],
                    [9.0, 6.0, 3.0, 8.0, 2.0, 5.0],
                ],
            ]
        ],
        dtype=torch.float64,
    )
    spatial_mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0, 1],
            [0, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    boxes = [(0, 3, 0, 4), (0, 3, 2, 6), (1, 4, 0, 4), (1, 4, 2, 6)]
    if reverse_order:
        boxes.reverse()

    reducer = StreamingSigmoidAttentionPoolingReducer(
        tau_init=0.43,
        learnable_temperature=False,
        accumulator_dtype=accumulator_dtype,
    )
    reducer.start_stream(4, 6, 1, 3, source.device, source.dtype)
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    accepted = torch.zeros_like(spatial_mask)
    owned = torch.zeros_like(spatial_mask)
    first_divergence = {
        name: None for name in ("running_m", "running_zhat", "running_shat")
    }
    tolerances = {
        "running_m": (0.0, 0.0),
        "running_zhat": (2e-15, 2e-15),
        "running_shat": (2e-14, 2e-15),
    }

    for tile_index, (y0, y1, x0, x1) in enumerate(boxes):
        newly_owned = ~owned[y0:y1, x0:x1]
        accepted[y0:y1, x0:x1] |= newly_owned & spatial_mask[y0:y1, x0:x1]
        owned[y0:y1, x0:x1] = True
        reducer.accumulate_stream_tile(
            source[..., y0:y1, x0:x1],
            y0,
            x0,
            sides,
            (y0, y1, x0, x1),
            user_mask=spatial_mask[y0:y1, x0:x1],
        )
        reference = _sigmoid_attention_state_reference(
            source, reducer.current_tau, accepted, accumulator_dtype
        )
        for name, expected in zip(first_divergence, reference):
            actual = getattr(reducer, name)
            atol, rtol = tolerances[name]
            if first_divergence[name] is None and not _state_matches(
                actual, expected, atol=atol, rtol=rtol
            ):
                first_divergence[name] = (tile_index, *_state_error(actual, expected))

    final_reference = _sigmoid_attention_state_reference(
        source, reducer.current_tau, spatial_mask, accumulator_dtype
    )
    diagnostics = []
    for name, expected in zip(first_divergence, final_reference):
        actual = getattr(reducer, name)
        absolute_error, relative_error = _state_error(actual, expected)
        diagnostics.append(
            f"{name}: first_divergence={first_divergence[name]}, "
            f"final_absolute_error={absolute_error:.17g}, "
            f"final_relative_error={relative_error:.17g}"
        )
    assert all(value is None for value in first_divergence.values()), "; ".join(
        diagnostics
    )


def test_fixed_input_parity_across_single_nonoverlapping_and_overlapping_tiles():
    """Locate the first tiling regime that leaves float64 round-off parity."""
    accumulator_dtype = torch.float64
    logits = -torch.tensor(
        [
            [
                [
                    [8, 1, 4, 9, 2, 7],
                    [3, 6, 2, 5, 8, 1],
                    [9, 4, 7, 3, 6, 2],
                    [2, 8, 1, 6, 4, 9],
                ],
                [
                    [2, 7, 5, 1, 8, 4],
                    [6, 3, 9, 2, 5, 7],
                    [1, 8, 4, 6, 2, 9],
                    [7, 2, 6, 4, 9, 3],
                ],
                [
                    [5, 9, 2, 7, 3, 6],
                    [8, 1, 4, 9, 6, 2],
                    [3, 7, 5, 1, 8, 4],
                    [9, 6, 3, 8, 2, 5],
                ],
            ]
        ],
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0, 1],
            [0, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    upstream = torch.tensor([[[[0.75]], [[-1.25]], [[2.0]]]], dtype=torch.float64)
    configurations = {
        "single_complete_tile": [(0, 4, 0, 6)],
        "multiple_nonoverlapping_tiles": [
            (0, 2, 0, 3),
            (0, 2, 3, 6),
            (2, 4, 0, 3),
            (2, 4, 3, 6),
        ],
        "multiple_overlapping_tiles": [
            (0, 3, 0, 4),
            (0, 3, 2, 6),
            (1, 4, 0, 4),
            (1, 4, 2, 6),
        ],
    }

    # Construct the dense baseline once.  Every tiled run below copies this
    # reducer, so weights (raw_tau), effective temperature, mask, logits,
    # accumulator dtype, and upstream gradient cannot drift between cases.
    dense_reducer = SigmoidAttentionPoolingReducer(
        tau_init=0.43, accumulator_dtype=accumulator_dtype
    )
    dense_logits = logits.clone().requires_grad_()
    dense_output = dense_reducer(dense_logits, mask=mask)
    dense_output.backward(upstream)
    dense_gradient = dense_logits.grad.detach().clone()
    state_reference = _sigmoid_attention_state_reference(
        logits, dense_reducer.current_tau, mask, accumulator_dtype
    )

    results = []
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    for name, boxes in configurations.items():
        streaming = dense_reducer.to_streaming()
        streaming.start_stream(4, 6, 1, 3, logits.device, logits.dtype)
        assembled_logits = torch.empty_like(logits)
        assigned = torch.zeros_like(mask)
        replay_tiles = []

        for box in boxes:
            y0, y1, x0, x1 = box
            tile = logits[..., y0:y1, x0:x1]
            valid_mask = ~assigned[y0:y1, x0:x1]
            valid_mask &= mask[y0:y1, x0:x1]
            # Record the exact overlap-safe ownership and destination used by
            # backward replay, independently of reducer internals.
            replay_tiles.append((box, valid_mask.clone()))
            assembled_logits[..., y0:y1, x0:x1] = tile
            assigned[y0:y1, x0:x1] = True
            streaming.accumulate_stream_tile(
                tile, y0, x0, sides, box, user_mask=mask[y0:y1, x0:x1]
            )

        output = streaming.finish_stream()
        replay_gradient = torch.zeros_like(logits)
        context = streaming.extra_state_for_backward()
        for (y0, y1, x0, x1), valid_mask in replay_tiles:
            replay_logits = logits[..., y0:y1, x0:x1].clone().requires_grad_()
            replay_output = streaming.reduce_tile_for_backward(
                replay_logits, valid_mask, context
            )
            replay_output.backward(upstream)
            replay_gradient[..., y0:y1, x0:x1] += replay_logits.grad

        actuals = {
            "reducer_input_logits": assembled_logits,
            "running_m": streaming.running_m,
            "running_zhat": streaming.running_zhat,
            "running_shat": streaming.running_shat,
            "final_output": output,
            "reducer_input_gradient": replay_gradient,
        }
        expected = {
            "reducer_input_logits": logits,
            "running_m": state_reference[0],
            "running_zhat": state_reference[1],
            "running_shat": state_reference[2],
            "final_output": dense_output.detach(),
            "reducer_input_gradient": dense_gradient,
        }
        errors = {
            key: _max_absolute_error(value, expected[key])
            for key, value in actuals.items()
        }
        outside_precision = [
            key
            for key, error in errors.items()
            if error > _precision_limit(expected[key])
        ]
        results.append((name, errors, outside_precision))

        assert (
            assigned.all()
        ), f"{name}: destination assignments leave output pixels uncovered"
        assert torch.equal(
            streaming._stream_seen_mask, assigned
        ), f"{name}: overlap-safe valid_mask ownership disagrees with destination assignments"

    first = next((result for result in results if result[2]), None)
    report = (
        "no configuration exceeded machine precision"
        if first is None
        else (
            f"first configuration beyond machine precision: {first[0]}; "
            f"metrics={first[2]}; errors={first[1]}"
        )
    )
    tolerances = {
        "reducer_input_logits": 0.0,
        "running_m": 0.0,
        "running_zhat": 2e-15,
        "running_shat": 2e-14,
        "final_output": 2e-15,
        "reducer_input_gradient": 2e-14,
    }
    failures = [
        f"{name}.{metric}={error:.17g} (limit={tolerances[metric]:.17g})"
        for name, errors, _ in results
        for metric, error in errors.items()
        if error > tolerances[metric]
    ]
    assert not failures, f"{report}; " + "; ".join(failures)


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
    expected_tau_grad = streaming.raw_tau.grad.detach().clone() if learnable else None
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


@pytest.mark.parametrize(
    ("input_dtype", "accumulator_dtype", "module_dtype"),
    [
        pytest.param(torch.float16, torch.float32, torch.float32, id="fp16-fp32"),
        pytest.param(
            torch.bfloat16, torch.float32, torch.float32, id="bf16-fp32"
        ),
        pytest.param(torch.float32, torch.float64, torch.float32, id="fp32-fp64"),
        pytest.param(
            torch.float32, torch.float64, torch.float64, id="fp64-module-state"
        ),
    ],
)
@pytest.mark.parametrize("learnable", [False, True], ids=["buffer", "parameter"])
def test_conversion_keeps_module_input_and_accumulator_dtypes_independent(
    input_dtype, accumulator_dtype, module_dtype, learnable
):
    if input_dtype == torch.bfloat16:
        try:
            torch.softmax(torch.zeros(2, dtype=input_dtype), dim=0)
        except RuntimeError:
            pytest.skip("bfloat16 softmax is not supported on this device")

    offline = SigmoidAttentionPoolingReducer(
        tau_init=0.37,
        learnable_temperature=learnable,
        accumulator_dtype=accumulator_dtype,
    ).to(dtype=module_dtype)
    source_raw_tau = offline.raw_tau.detach().clone()
    source_tau = offline.current_tau.detach().clone()
    x = torch.linspace(-2, 2, 24, dtype=input_dtype).reshape(1, 2, 3, 4)
    assert offline(x).dtype == input_dtype

    streaming = offline.to_streaming()
    assert streaming.accumulator_dtype == accumulator_dtype
    assert streaming.raw_tau.dtype == module_dtype
    assert torch.equal(streaming.raw_tau, source_raw_tau)
    assert torch.equal(streaming.current_tau, source_tau)
    assert isinstance(streaming.raw_tau, torch.nn.Parameter) == learnable

    streaming.start_stream(3, 4, 1, 2, x.device, x.dtype)
    sides = SimpleNamespace(top=False, left=False, right=False, bottom=False)
    streaming.accumulate_stream_tile(x, 0, 0, sides, (0, 3, 0, 4))
    output = streaming.finish_stream()

    assert streaming.running_m.dtype == accumulator_dtype
    assert streaming.running_zhat.dtype == accumulator_dtype
    assert streaming.running_shat.dtype == accumulator_dtype
    assert output.dtype == input_dtype
    assert streaming.raw_tau.dtype == module_dtype

    restored = streaming.to_reducer()
    assert restored.accumulator_dtype == accumulator_dtype
    assert restored.raw_tau.dtype == module_dtype
    assert torch.equal(restored.raw_tau, source_raw_tau)
    assert torch.equal(restored.current_tau, source_tau)
    assert isinstance(restored.raw_tau, torch.nn.Parameter) == learnable
