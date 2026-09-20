import copy

import pytest
import torch
import torch.nn as nn

from lightstream.core.reducer import SoftmaxAttentionReducer, StreamingSoftmaxAttentionReducer
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
