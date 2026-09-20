import pytest
import torch
import torch.nn.functional as F

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.layers import StreamingMerge
from lightstream.core.layers.streamingneighborhoodattention import NeighborhoodAttention2D
from lightstream.core.scnn.scnn import StreamingCNN


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


class _LocalMeanAttention(torch.nn.Module):
    """Small NHWC attention stand-in with the same kernel-halo contract."""

    kernel_size = (7, 7)
    dilation = (1, 1)

    def __init__(self, channels):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.linspace(0.5, 1.5, channels))

    def forward(self, value):
        nchw = value.permute(0, 3, 1, 2)
        local_mean = F.avg_pool2d(nchw, self.kernel_size, stride=1, padding=3)
        return (local_mean * self.scale[None, :, None, None]).permute(0, 2, 3, 1)


class _AttentionResidual(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = NeighborhoodAttention2D(attention=_LocalMeanAttention(channels))
        self.merge = StreamingMerge("add")

    def forward(self, value):
        return self.merge(value, self.attention(value))


def test_streaming_attention_residual_shifted_tiles_match_complete_input_gradient():
    """Residual identity ownership must not discard attention halo gradients."""

    torch.manual_seed(20260920)
    channels = 3
    module = _AttentionResidual(channels)
    full_module = _AttentionResidual(channels)
    full_module.load_state_dict(module.state_dict())
    streaming = StreamingCNN(module, tile_shape=(1, channels, 11, 13), copy_to_gpu=True)

    # Kernel-7 leaves 5x7 owned queries per regular tile.  Both 16 and 19
    # require the final tile to shift backwards and overlap its predecessor.
    full_input = torch.randn(1, channels, 16, 19, requires_grad=True)
    streamed_input = full_input.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(full_input)

    full_module(full_input).backward(upstream)
    streaming(streamed_input)
    streaming.backward(streamed_input, upstream)

    torch.testing.assert_close(streamed_input.grad, full_input.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(
        streaming.stream_module.attention.attention.scale.grad,
        full_module.attention.attention.scale.grad,
        rtol=2e-4,
        atol=2e-5,
    )
