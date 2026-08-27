import torch

from lightstream.core.reducer import SoftmaxAttentionReducer, StreamingSoftmaxAttentionReducer


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
