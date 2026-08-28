import pytest
import torch

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.scnn.streamingmerge import StreamingMerge


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("add", torch.add), ("multiply", torch.mul)],
)
def test_streaming_merge_has_exact_eager_semantics(mode, expected):
    a = torch.randn(2, 3, 4, 5, requires_grad=True)
    b = torch.randn(2, 3, 4, 5, requires_grad=True)

    assert torch.equal(StreamingMerge(mode)(a, b), expected(a, b))


def test_streaming_merge_multiply_retains_pytorch_overflow_semantics():
    a = torch.full((1, 1, 2, 2), 1e30)
    b = torch.full((1, 1, 2, 2), 1e30)

    result = StreamingMerge("multiply")(a, b)

    assert torch.equal(result, a * b)
    assert torch.isinf(result).all()


def test_streaming_merge_multiply_statistics_mode_preserves_finite_gradients():
    merge = StreamingMerge("multiply")
    merge._streaming_statistics_mode = True
    a = torch.full((1, 1, 2, 2), 1e30, requires_grad=True)
    b = torch.full((1, 1, 2, 2), 2e30, requires_grad=True)

    result = merge(a, b)
    result.sum().backward()

    assert torch.isfinite(result).all()
    assert a.grad is not None and torch.isfinite(a.grad).all() and torch.count_nonzero(a.grad)
    assert b.grad is not None and torch.isfinite(b.grad).all() and torch.count_nonzero(b.grad)


def test_streaming_merge_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        StreamingMerge("subtract")


def test_streaming_merge_requires_matching_spatial_shapes():
    with pytest.raises(ValueError, match="compatible spatial shapes"):
        StreamingMerge("add")(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 4, 4))


def test_streaming_constructor_preserves_merge_boundaries():
    constructor = StreamingConstructor(torch.nn.Linear(2, 2), tile_size=4)

    assert StreamingMerge in constructor.keep_modules
