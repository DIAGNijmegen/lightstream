import sys
import types

import pytest
import torch

from lightstream.core.scnn import ChannelLayerNorm, StreamingChannelLayerNorm
from lightstream.core.scnn.streaminglayernorm import ChannelLayerNorm as ImportedChannelLayerNorm


def test_channel_layer_norm_matches_nhwc_layer_norm():
    torch.manual_seed(7)
    module = ChannelLayerNorm(3, eps=1e-6, elementwise_affine=True)
    reference = torch.nn.LayerNorm(3, eps=1e-6, elementwise_affine=True)
    reference.load_state_dict(module.norm.state_dict())

    x = torch.randn(2, 3, 5, 7)
    expected = reference(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    torch.testing.assert_close(module(x), expected)


def test_channel_layer_norm_rejects_non_4d_input():
    module = ChannelLayerNorm(3)

    with pytest.raises(ValueError, match="expects 4D NCHW input"):
        module(torch.randn(2, 3, 5))


def test_channel_layer_norm_rejects_channel_mismatch():
    module = ChannelLayerNorm(3)

    with pytest.raises(ValueError, match="expected 3 channels"):
        module(torch.randn(2, 4, 5, 7))


def test_channel_layer_norm_is_public_from_scnn_package():
    assert ImportedChannelLayerNorm is ChannelLayerNorm
    assert StreamingChannelLayerNorm.__name__ == "StreamingChannelLayerNorm"


def test_constructor_considers_only_channel_layer_norm_streamable(monkeypatch):
    # The constructor imports StreamingCNN, whose module imports numpy for full
    # streaming execution. This test only needs the constructor keep-list, so a
    # minimal import-time stub keeps the unit test focused when numpy is not
    # installed in the local environment.
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    from lightstream.core.constructor import StreamingConstructor

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 1),
        ChannelLayerNorm(3),
        torch.nn.LayerNorm(3),
    )
    constructor = StreamingConstructor(model, tile_size=32, verbose=False, statistics_on_cpu=True)

    assert ChannelLayerNorm in constructor.keep_modules
    assert torch.nn.LayerNorm not in constructor.keep_modules


def test_streaming_statistics_hooks_include_channel_layer_norm(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))

    from lightstream.core.scnn.scnn import StreamingCNN
    from lightstream.core.scnn.utils import Lost

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
        ChannelLayerNorm(3),
        torch.nn.Conv2d(3, 3, kernel_size=1, bias=False),
    ).eval()

    scnn = StreamingCNN(
        model,
        tile_shape=(1, 3, 6, 6),
        verbose=False,
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
    )

    norm = scnn.stream_module[1]
    stats = scnn._module_stats[norm]

    assert stats["stride"].tolist() == [1, 1, 1]
    assert stats["lost"] == Lost(1, 1, 1, 1)
    assert stats["grad_lost"] == Lost(1, 1, 1, 1)
    assert stats["output_stride"].tolist() == [1, 1, 1]


def test_streaming_channel_layer_norm_conversion_preserves_parameters_and_metadata():
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm

    module = ChannelLayerNorm(3, eps=1e-4, elementwise_affine=True).to(dtype=torch.float64)
    module.norm.weight.data.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    module.norm.bias.data.copy_(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))
    module.norm.weight.requires_grad = False
    module.norm.bias.requires_grad = True

    streaming = StreamingChannelLayerNorm.from_channel_layer_norm(module)

    assert streaming.num_channels == module.num_channels
    assert streaming.eps == module.norm.eps
    assert streaming.elementwise_affine == module.norm.elementwise_affine
    assert streaming.weight.dtype == module.norm.weight.dtype
    assert streaming.weight.device == module.norm.weight.device
    assert streaming.weight.requires_grad == module.norm.weight.requires_grad
    assert streaming.bias.requires_grad == module.norm.bias.requires_grad
    torch.testing.assert_close(streaming.weight, module.norm.weight)
    torch.testing.assert_close(streaming.bias, module.norm.bias)

    restored = streaming.to_channel_layer_norm()
    assert restored.num_channels == module.num_channels
    assert restored.norm.eps == module.norm.eps
    assert restored.norm.elementwise_affine == module.norm.elementwise_affine
    assert restored.norm.weight.requires_grad == module.norm.weight.requires_grad
    assert restored.norm.bias.requires_grad == module.norm.bias.requires_grad
    torch.testing.assert_close(restored.norm.weight, module.norm.weight)
    torch.testing.assert_close(restored.norm.bias, module.norm.bias)


def test_scnn_converts_nested_channel_layer_norm_and_transfers_stats():
    from lightstream.core.scnn.scnn import StreamingCNN
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm
    from lightstream.core.scnn.utils import Lost

    norm = ChannelLayerNorm(3)
    model = torch.nn.Sequential(torch.nn.Sequential(norm))
    stats = {
        "grad_lost": Lost(2, 3, 4, 5),
        "output_stride": torch.tensor([1, 2, 2]),
    }
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn._module_stats = {norm: stats}
    scnn._streaming_reducers = []

    converted = scnn._convert_modules_for_streaming(model)

    streaming_norm = converted[0][0]
    assert isinstance(streaming_norm, StreamingChannelLayerNorm)
    assert streaming_norm.grad_lost == stats["grad_lost"]
    torch.testing.assert_close(streaming_norm.output_stride, stats["output_stride"])
    assert scnn._module_stats == {streaming_norm: stats}


def test_scnn_resets_streaming_channel_layer_norm_and_preserves_stats():
    from lightstream.core.scnn.scnn import StreamingCNN
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm
    from lightstream.core.scnn.utils import Lost

    streaming_norm = StreamingChannelLayerNorm(3)
    model = torch.nn.Sequential(streaming_norm)
    stats = {
        "grad_lost": Lost(1, 1, 1, 1),
        "output_stride": torch.tensor([1, 4, 4]),
    }
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn._module_stats = {streaming_norm: stats}

    restored = scnn._reset_converted_modules(model)

    norm = restored[0]
    assert isinstance(norm, ChannelLayerNorm)
    assert scnn._module_stats == {norm: stats}


def test_streaming_channel_layer_norm_matches_channel_layer_norm_forward_and_backward():
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm

    torch.manual_seed(11)
    module = ChannelLayerNorm(4, eps=1e-5, elementwise_affine=True)
    streaming = StreamingChannelLayerNorm.from_channel_layer_norm(module)

    x = torch.randn(2, 4, 3, 5, requires_grad=True)
    x_streaming = x.detach().clone().requires_grad_(True)
    grad = torch.randn(2, 4, 3, 5)

    module(x).backward(grad)
    streaming(x_streaming).backward(grad)

    torch.testing.assert_close(x_streaming.grad, x.grad)
    torch.testing.assert_close(streaming.weight.grad, module.norm.weight.grad)
    torch.testing.assert_close(streaming.bias.grad, module.norm.bias.grad)


def test_streaming_channel_layer_norm_affine_grads_use_only_unique_valid_region():
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm
    from lightstream.core.scnn.utils import Box, Lost, Sides

    torch.manual_seed(13)
    streaming = StreamingChannelLayerNorm(3, eps=1e-5, elementwise_affine=True)
    streaming.grad_lost = Lost(top=1, left=1, bottom=1, right=0)
    streaming.input_loc = Box(y=0, height=4, x=0, width=5, sides=Sides(left=1, top=1, right=0, bottom=0))
    streaming.output_stride = torch.tensor([1, 1, 1])

    x = torch.randn(2, 3, 4, 5, requires_grad=True)
    grad = torch.randn(2, 3, 4, 5)
    streaming(x).backward(grad)

    with torch.no_grad():
        centered = x - x.mean(dim=1, keepdim=True)
        x_hat = centered * torch.rsqrt(centered.pow(2).mean(dim=1, keepdim=True) + streaming.eps)
        expected_grad_weight = (grad[:, :, :3, :] * x_hat[:, :, :3, :]).sum(dim=(0, 2, 3))
        expected_grad_bias = grad[:, :, :3, :].sum(dim=(0, 2, 3))

    torch.testing.assert_close(streaming.weight.grad, expected_grad_weight)
    torch.testing.assert_close(streaming.bias.grad, expected_grad_bias)


def test_streaming_channel_layer_norm_without_affine_backpropagates_input():
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm

    torch.manual_seed(17)
    module = ChannelLayerNorm(3, elementwise_affine=False)
    streaming = StreamingChannelLayerNorm.from_channel_layer_norm(module)

    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    x_streaming = x.detach().clone().requires_grad_(True)
    grad = torch.randn(2, 3, 4, 4)

    module(x).backward(grad)
    streaming(x_streaming).backward(grad)

    assert streaming.weight is None
    assert streaming.bias is None
    torch.testing.assert_close(x_streaming.grad, x.grad)


def test_backward_streaming_module_predicate_includes_existing_and_layer_norm_types(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))

    from lightstream.core.scnn.scnn import _is_backward_streaming_module
    from lightstream.core.scnn.streamingconv import StreamingConv2d
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm
    from lightstream.core.scnn.streamingupsample import StreamingUpsample2d

    assert _is_backward_streaming_module(StreamingConv2d(3, 3, kernel_size=1))
    assert _is_backward_streaming_module(StreamingUpsample2d(scale_factor=2.0, mode="bilinear"))
    assert _is_backward_streaming_module(StreamingChannelLayerNorm(3))
    assert not _is_backward_streaming_module(torch.nn.ReLU())
