import copy

import pytest
import torch
import torch.nn as nn

from lightstream.core.reducer import (
    AttentionGeMReducer, SoftmaxAttentionReducer, StreamingSoftmaxAttentionReducer,
)
from lightstream.core.scnn.scnn import StreamingCNN


class _PointwiseAttentionHead(nn.Module):
    """Smallest network which exercises both reducer producer branches."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Conv2d(3, 1, kernel_size=1, dtype=torch.float64)
        self.att_logits = nn.Conv2d(3, 1, kernel_size=1, dtype=torch.float64)
        self.reducer = SoftmaxAttentionReducer(accumulator_dtype=torch.float64)

    def forward(self, features):
        return self.reducer(self.classifier(features), self.att_logits(features))


@pytest.mark.parametrize("stopgrad", [False, True])
def test_stopgrad_keeps_value_gradient_and_controls_attention_gradient(stopgrad):
    values = torch.tensor([[[[-2.0, 1.0], [3.0, 4.0]]]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([[[[0.3, -0.2], [0.7, 1.1]]]], dtype=torch.float64, requires_grad=True)
    reducer = SoftmaxAttentionReducer(stopgrad_attention=stopgrad, accumulator_dtype=torch.float64)
    actual = reducer(values, logits)
    weights = torch.softmax(logits.detach().flatten(2), dim=-1).view_as(logits)
    expected = (weights * values.detach()).sum((-2, -1), keepdim=True)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(values.grad, weights)
    if stopgrad:
        assert logits.grad is None
    else:
        torch.testing.assert_close(logits.grad, weights * (values.detach() - expected))

    streamed = reducer.to_streaming()
    assert streamed.stopgrad_attention is stopgrad
    assert streamed.to_reducer().stopgrad_attention is stopgrad


@pytest.mark.parametrize("stopgrad", [False, True])
def test_stopgrad_replay_matches_full_frame(stopgrad):
    torch.manual_seed(103)
    values = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True, True], [True, True, False, True], [False, True, True, True]])
    reducer = SoftmaxAttentionReducer(stopgrad_attention=stopgrad, accumulator_dtype=torch.float64)
    upstream = torch.randn(1, 2, 1, 1, dtype=torch.float64)
    torch.autograd.backward(reducer(values, logits, mask=mask), upstream)

    streaming = reducer.to_streaming()
    streaming.accumulate_valid_tile((values.detach(), logits.detach()), mask)
    replay_values = values.detach().clone().requires_grad_()
    replay_logits = logits.detach().clone().requires_grad_()
    replay = streaming.reduce_tile_for_backward(
        (replay_values, replay_logits), mask, streaming.extra_state_for_backward()
    )
    torch.autograd.backward(replay, upstream)
    torch.testing.assert_close(replay_values.grad, values.grad, rtol=1e-10, atol=1e-12)
    if stopgrad:
        assert logits.grad is None and replay_logits.grad is None
    else:
        torch.testing.assert_close(replay_logits.grad, logits.grad, rtol=1e-10, atol=1e-12)


def test_same_tensor_as_values_and_attention_only_keeps_value_path():
    logits = torch.tensor([[[[-1.0, 2.0], [0.5, 3.0]]]], dtype=torch.float64, requires_grad=True)
    SoftmaxAttentionReducer(stopgrad_attention=True)(logits, logits).sum().backward()
    expected = torch.softmax(logits.detach().flatten(2), dim=-1).view_as(logits)
    torch.testing.assert_close(logits.grad, expected)


def test_streaming_reducers_keep_distinct_payloads_for_shared_producers():
    values = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    gem = AttentionGeMReducer().to_streaming()
    softmax = SoftmaxAttentionReducer().to_streaming()
    gem_payload = gem(values, logits)
    softmax_payload = softmax(values, logits)
    for position, source in enumerate((values, logits)):
        assert gem_payload[position] is not softmax_payload[position]
        assert gem_payload[position].data_ptr() == source.data_ptr()
        assert softmax_payload[position].data_ptr() == source.data_ptr()
        assert gem_payload[position].requires_grad and softmax_payload[position].requires_grad
    assert gem._last_inputs is gem_payload and softmax._last_inputs is softmax_payload


class _TwoIndependentReducers(nn.Module):
    def __init__(self):
        super().__init__()
        self.values = nn.Conv2d(3, 2, 1, dtype=torch.float64)
        self.attention = nn.Conv2d(3, 1, 1, dtype=torch.float64)
        self.gem = AttentionGeMReducer(r_init=1.5, eps=1e-9,
                                       accumulator_dtype=torch.float64, stopgrad_attention=True)
        self.softmax = SoftmaxAttentionReducer(accumulator_dtype=torch.float64,
                                                stopgrad_attention=False)
        with torch.no_grad():
            self.values.bias.fill_(1.0)

    def forward(self, features):
        values, logits = self.values(features), self.attention(features)
        return self.gem(values, logits), self.softmax(values, logits)


def test_independent_reducers_share_producers_without_losing_tile_gradients():
    torch.manual_seed(107)
    features = torch.randn(1, 3, 5, 7, dtype=torch.float64) * 0.05
    reference = _TwoIndependentReducers()
    streamed_model = copy.deepcopy(reference)
    upstream = (torch.randn(1, 2, 1, 1, dtype=torch.float64),
                torch.randn(1, 2, 1, 1, dtype=torch.float64))
    full_features = features.clone().requires_grad_()
    torch.autograd.backward(reference(full_features), upstream)
    streamed = StreamingCNN(
        streamed_model, tile_shape=(1, 3, 4, 4), deterministic=True,
        saliency=False, copy_to_gpu=False, statistics_on_cpu=False,
        normalize_on_gpu=False,
    )
    streamed_features = features.clone().requires_grad_()
    for actual, expected in zip(streamed(streamed_features), reference(features)):
        torch.testing.assert_close(actual, expected)
    streamed.backward(streamed_features, upstream)
    torch.testing.assert_close(streamed_features.grad, full_features.grad, rtol=1e-9, atol=1e-11)
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(dict(streamed_model.named_parameters())[name].grad,
                                   parameter.grad, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("stopgrad", [False, True])
def test_streamed_stopgrad_preserves_attention_branch_statistics(stopgrad):
    torch.manual_seed(109)
    features = torch.randn(1, 3, 5, 7, dtype=torch.float64)
    upstream = torch.tensor([[[[1.25]]]], dtype=torch.float64)
    reference = _PointwiseAttentionHead()
    reference.reducer.stopgrad_attention = stopgrad
    streamed_model = copy.deepcopy(reference)
    full_features = features.clone().requires_grad_()
    torch.autograd.backward(reference(full_features), upstream)

    streamed = StreamingCNN(
        streamed_model, tile_shape=(1, 3, 4, 4), deterministic=True,
        saliency=False, copy_to_gpu=False, statistics_on_cpu=False,
        normalize_on_gpu=False,
    )
    streamed_features = features.clone().requires_grad_()
    torch.testing.assert_close(streamed(streamed_features), reference(features))
    streamed.backward(streamed_features, upstream)
    torch.testing.assert_close(streamed_features.grad, full_features.grad, rtol=1e-10, atol=1e-12)
    for name, ref_param in reference.named_parameters():
        actual = dict(streamed_model.named_parameters())[name].grad
        if stopgrad and name.startswith("att_logits."):
            assert actual is None and ref_param.grad is None
        else:
            torch.testing.assert_close(actual, ref_param.grad, rtol=1e-10, atol=1e-12)


def test_matches_reference_for_all_attention_shapes_and_signed_values():
    torch.manual_seed(4)
    values = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    shared = torch.randn(2, 4, 5, dtype=torch.float64)
    for logits in (shared, shared[:, None], shared[:, None].expand(-1, 3, -1, -1)):
        actual = SoftmaxAttentionReducer()(values, logits)
        expected = (values * torch.softmax(shared.flatten(1), dim=1).view(2, 1, 4, 5)).sum((-2, -1), keepdim=True)
        torch.testing.assert_close(actual, expected)


def test_uniform_logits_are_mean_and_values_are_not_transformed():
    values = torch.tensor([[[[-3.0, 1.0], [2.0, 8.0]]]])
    result = SoftmaxAttentionReducer()(values, torch.zeros(1, 2, 2))
    torch.testing.assert_close(result, values.mean((-2, -1), keepdim=True))


def test_mask_resize_and_fully_masked_sample():
    values = torch.arange(8.0).view(2, 1, 2, 2)
    logits = torch.tensor([[[1000.0, -1000.0], [0.0, 1.0]]] * 2)
    mask = torch.tensor([[[1]], [[0]]])
    result = SoftmaxAttentionReducer(mask_resize=True)(values, logits, mask=mask)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[1], torch.zeros_like(result[1]))
    torch.testing.assert_close(result[0], values[0, :, :1, :1])


def test_low_precision_output_dtype_and_gradients():
    values = torch.randn(1, 2, 3, 3, dtype=torch.float16)
    logits = torch.randn(1, 3, 3, dtype=torch.float16)
    assert SoftmaxAttentionReducer()(values, logits).dtype == torch.float16
    v = values.float().requires_grad_()
    a = logits.float().requires_grad_()
    SoftmaxAttentionReducer()(v, a).sum().backward()
    assert torch.isfinite(v.grad).all() and torch.isfinite(a.grad).all()


def test_conversion():
    offline = SoftmaxAttentionReducer(accumulator_dtype=torch.float64, mask_resize=True)
    streaming = offline.to_streaming()
    assert isinstance(streaming, StreamingSoftmaxAttentionReducer)
    restored = streaming.to_reducer()
    assert restored.accumulator_dtype == torch.float64 and restored.mask_resize


@pytest.mark.parametrize("height,width", [(8, 8), (5, 7)])
def test_streamed_backward_owns_each_reducer_position_once(height, width):
    """Pointwise parameter gradients must not count shifted-tile overlap twice."""
    torch.manual_seed(83)
    features = torch.randn(1, 3, height, width, dtype=torch.float64)
    upstream = torch.tensor([[[[1.75]]]], dtype=torch.float64)
    reference = _PointwiseAttentionHead()
    streamed_model = copy.deepcopy(reference)

    full_features = features.clone().requires_grad_()
    torch.autograd.backward(reference(full_features), upstream)

    streamed = StreamingCNN(
        streamed_model,
        tile_shape=(1, 3, 4, 4),
        deterministic=True,
        saliency=False,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
    )
    streamed.debug_reducer_replay = True
    streamed_features = features.clone().requires_grad_()
    stream_output = streamed(streamed_features)
    torch.testing.assert_close(stream_output, reference(features))
    streamed.backward(streamed_features, upstream)

    # The reducer input and both one-by-one producer convolutions agree with a
    # single full-frame autograd graph.
    torch.testing.assert_close(streamed_features.grad, full_features.grad, rtol=1e-10, atol=1e-12)
    for name in (
        "classifier.weight",
        "classifier.bias",
        "att_logits.weight",
        "att_logits.bias",
    ):
        torch.testing.assert_close(
            dict(streamed_model.named_parameters())[name].grad,
            dict(reference.named_parameters())[name].grad,
            rtol=1e-10,
            atol=1e-12,
        )

    # A one-channel classifier's additive bias derivative is the upstream
    # scalar because the global softmax weights sum to exactly one.
    torch.testing.assert_close(streamed_model.classifier.bias.grad, upstream.flatten())

    reducer = streamed_model.reducer
    coordinates = torch.cat(reducer._backward_replay_regions, dim=0)
    assert coordinates.shape == (height * width, 2)
    assert torch.unique(coordinates, dim=0).shape[0] == height * width
    assert set(map(tuple, coordinates.tolist())) == set(
        map(tuple, torch.cartesian_prod(torch.arange(height), torch.arange(width)).tolist())
    )

    if (height, width) == (5, 7):
        # Both axes end in shifted tiles.  Every recorded tile owns a disjoint
        # global coordinate set, including the final row/column tiles.
        for index, region in enumerate(reducer._backward_replay_regions):
            earlier = reducer._backward_replay_regions[:index]
            if earlier and region.numel():
                old = torch.cat(earlier, dim=0)
                assert not (region[:, None, :] == old[None, :, :]).all(dim=-1).any()
