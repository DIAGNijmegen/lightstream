import sys
import types

import pytest
import torch

from lightstream.core.scnn import ChannelLayerNorm, StreamingChannelLayerNorm
from lightstream.core.scnn.streaminglayernorm import ChannelLayerNorm as ImportedChannelLayerNorm


def _channel_layer_norm_affine_keys(module: torch.nn.Module) -> set[str]:
    return {key for key in module.state_dict() if key.endswith(("norm.weight", "norm.bias"))}


def _assert_channel_layer_norm_state_keys_compatible(
    original: torch.nn.Module, converted: torch.nn.Module, *, prefix: str = ""
) -> None:
    expected_keys = {f"{prefix}norm.weight", f"{prefix}norm.bias"}
    assert _channel_layer_norm_affine_keys(original) == expected_keys
    assert _channel_layer_norm_affine_keys(converted) == expected_keys


def _assert_no_channel_layer_norm_affine_keys(module: torch.nn.Module) -> None:
    assert _channel_layer_norm_affine_keys(module) == set()


def test_channel_layer_norm_matches_nhwc_layer_norm():
    torch.manual_seed(7)
    module = ChannelLayerNorm(3, eps=1e-6, elementwise_affine=True)
    reference = torch.nn.LayerNorm(3, eps=1e-6, elementwise_affine=True)
    reference.load_state_dict(module.norm.state_dict())

    x = torch.randn(2, 3, 5, 7)
    expected = reference(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    torch.testing.assert_close(module(x), expected)


def test_channel_layer_norm_statistics_passthrough_returns_input_identity():
    torch.manual_seed(8)
    module = ChannelLayerNorm(3, eps=1e-6, elementwise_affine=True)
    x = torch.randn(2, 3, 5, 7)

    normal_output = module(x)
    module._streaming_statistics_passthrough = True
    passthrough_output = module(x)
    module._streaming_statistics_passthrough = False

    assert passthrough_output is x
    torch.testing.assert_close(passthrough_output, x)
    assert not torch.allclose(normal_output, x)
    torch.testing.assert_close(module(x), normal_output)


def test_scnn_toggles_channel_layer_norm_statistics_passthrough(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))

    from lightstream.core.scnn.scnn import StreamingCNN

    norm = ChannelLayerNorm(3)
    model = torch.nn.Sequential(norm)
    scnn = StreamingCNN.__new__(StreamingCNN)
    torch.nn.Module.__init__(scnn)
    scnn.stream_module = model

    scnn._set_channel_layer_norm_statistics_passthrough(True)
    assert norm._streaming_statistics_passthrough is True

    scnn._set_channel_layer_norm_statistics_passthrough(False)
    assert norm._streaming_statistics_passthrough is False


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


def test_convert_to_identity_skips_children_of_kept_channel_layer_norm(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    from lightstream.core.constructor import StreamingConstructor

    model = torch.nn.Sequential(
        torch.nn.Sequential(
            ChannelLayerNorm(3, eps=1e-6),
            torch.nn.ReLU(),
        ),
    )
    constructor = StreamingConstructor(model, tile_size=32, verbose=False, statistics_on_cpu=True)

    constructor.convert_to_identity(model)

    assert isinstance(model[0][0], ChannelLayerNorm)
    assert isinstance(model[0][0].norm, torch.nn.LayerNorm)
    assert model[0][0].norm.eps == 1e-6
    assert isinstance(model[0][1], torch.nn.Identity)


def test_constant_statistics_setup_does_not_reinitialize_channel_layer_norm(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))

    from lightstream.core.scnn.scnn import StreamingCNN

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(3),
        ChannelLayerNorm(3),
    ).eval()
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([2.0, 3.0, 4.0]))
        model[1].bias.copy_(torch.tensor([0.5, 0.25, -0.5]))
        model[2].norm.weight.copy_(torch.tensor([1.5, 2.5, 3.5]))
        model[2].norm.bias.copy_(torch.tensor([-1.0, 0.0, 1.0]))

    scnn = StreamingCNN.__new__(StreamingCNN)
    torch.nn.Module.__init__(scnn)
    scnn.stream_module = model

    StreamingCNN._reset_parameters_to_constant(scnn)

    torch.testing.assert_close(model[1].weight, torch.ones(3))
    torch.testing.assert_close(model[1].bias, torch.zeros(3))
    assert model[1].training is False
    torch.testing.assert_close(model[2].norm.weight, torch.tensor([1.5, 2.5, 3.5]))
    torch.testing.assert_close(model[2].norm.bias, torch.tensor([-1.0, 0.0, 1.0]))


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

    _assert_channel_layer_norm_state_keys_compatible(module, streaming)
    assert set(streaming.state_dict()) == {"norm.weight", "norm.bias"}
    assert set(dict(streaming.named_parameters())) == {"norm.weight", "norm.bias"}

    assert streaming.num_channels == module.num_channels
    assert streaming.eps == module.eps
    assert streaming.elementwise_affine == module.elementwise_affine
    assert streaming.weight.dtype == module.norm.weight.dtype
    assert streaming.weight.device == module.norm.weight.device
    assert streaming.weight.requires_grad == module.norm.weight.requires_grad
    assert streaming.bias.requires_grad == module.norm.bias.requires_grad
    torch.testing.assert_close(streaming.weight, module.norm.weight)
    torch.testing.assert_close(streaming.bias, module.norm.bias)

    restored = streaming.to_channel_layer_norm()
    _assert_channel_layer_norm_state_keys_compatible(module, restored)
    assert restored.num_channels == module.num_channels
    assert restored.eps == module.eps
    assert restored.elementwise_affine == module.elementwise_affine
    assert restored.norm.eps == module.eps
    assert restored.norm.elementwise_affine == module.elementwise_affine
    assert restored.norm.weight.requires_grad == module.norm.weight.requires_grad
    assert restored.norm.bias.requires_grad == module.norm.bias.requires_grad
    torch.testing.assert_close(restored.norm.weight, module.norm.weight)
    torch.testing.assert_close(restored.norm.bias, module.norm.bias)


def test_channel_layer_norm_stores_constructor_metadata():
    module = ChannelLayerNorm(5, eps=1e-3, elementwise_affine=False)

    assert module.num_channels == 5
    assert module.eps == 1e-3
    assert module.elementwise_affine is False


def test_streaming_channel_layer_norm_conversion_uses_channel_layer_norm_metadata():
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm

    module = ChannelLayerNorm(3, eps=1e-4, elementwise_affine=True)
    module.norm = torch.nn.LayerNorm(3, eps=1e-2, elementwise_affine=True)

    streaming = StreamingChannelLayerNorm.from_channel_layer_norm(module)

    _assert_channel_layer_norm_state_keys_compatible(module, streaming)
    assert set(streaming.state_dict()) == {"norm.weight", "norm.bias"}
    assert set(dict(streaming.named_parameters())) == {"norm.weight", "norm.bias"}

    assert streaming.num_channels == module.num_channels
    assert streaming.eps == module.eps
    assert streaming.elementwise_affine == module.elementwise_affine


def test_streaming_channel_layer_norm_conversion_rejects_replaced_norm():
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm

    module = ChannelLayerNorm(3)
    module.norm = torch.nn.Identity()

    with pytest.raises(TypeError, match="module.norm to be nn.LayerNorm"):
        StreamingChannelLayerNorm.from_channel_layer_norm(module)

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
    _assert_channel_layer_norm_state_keys_compatible(module, streaming)

    x = torch.randn(2, 4, 3, 5, requires_grad=True)
    x_streaming = x.detach().clone().requires_grad_(True)
    grad = torch.randn(2, 4, 3, 5)

    module(x).backward(grad)
    streaming(x_streaming).backward(grad)

    torch.testing.assert_close(x_streaming.grad, x.grad)
    reference_grads = {name: param.grad for name, param in module.named_parameters()}
    streaming_grads = {name: param.grad for name, param in streaming.named_parameters()}
    assert streaming_grads.keys() == reference_grads.keys() == {"norm.weight", "norm.bias"}
    for name in reference_grads:
        torch.testing.assert_close(streaming_grads[name], reference_grads[name])


def test_streaming_channel_layer_norm_affine_grads_use_only_unique_valid_region():
    from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm
    from lightstream.core.scnn.utils import Box, Lost, Sides

    torch.manual_seed(13)
    streaming = StreamingChannelLayerNorm(3, eps=1e-5, elementwise_affine=True)
    assert set(streaming.state_dict()) == {"norm.weight", "norm.bias"}
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

    _assert_no_channel_layer_norm_affine_keys(module)
    _assert_no_channel_layer_norm_affine_keys(streaming)
    assert set(dict(module.named_parameters())) == set()
    assert set(dict(streaming.named_parameters())) == set()

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


class SmallChannelLayerNormNet(torch.nn.Module):
    def __init__(self, elementwise_affine: bool = True):
        super().__init__()
        self.upstream = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
        self.norm = ChannelLayerNorm(4, eps=1e-5, elementwise_affine=elementwise_affine)
        self.activation = torch.nn.GELU()
        self.downstream = torch.nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upstream(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.downstream(x)


def _make_channel_norm_scnn(model: torch.nn.Module, *, tile_size: int, saliency: bool = False):
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["numpy"] = types.ModuleType("numpy")
    from lightstream.core.scnn.scnn import StreamingCNN

    return StreamingCNN(
        model,
        tile_shape=(1, 3, tile_size, tile_size),
        verbose=False,
        deterministic=True,
        saliency=saliency,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
    )


def _assert_channel_norm_scnn_parity(elementwise_affine: bool):
    torch.manual_seed(202)
    model = SmallChannelLayerNormNet(elementwise_affine=elementwise_affine).eval()
    reference = SmallChannelLayerNormNet(elementwise_affine=elementwise_affine).eval()
    reference.load_state_dict(model.state_dict())

    # Odd image dimensions with this tile size make the SCNN pass use overlapping
    # tiles; this is the case that would double-count affine gradients if
    # seen_indices tracking regressed.
    image = torch.randn(1, 3, 13, 11)
    upstream_grad = torch.randn(1, 2, 13, 11)

    ref_image = image.detach().clone().requires_grad_(True)
    ref_output = reference(ref_image)
    torch.autograd.backward(ref_output, upstream_grad)

    scnn = _make_channel_norm_scnn(model, tile_size=8)
    assert isinstance(scnn.stream_module.norm, StreamingChannelLayerNorm)
    if elementwise_affine:
        _assert_channel_layer_norm_state_keys_compatible(reference, scnn.stream_module, prefix="norm.")
    else:
        _assert_no_channel_layer_norm_affine_keys(reference)
        _assert_no_channel_layer_norm_affine_keys(scnn.stream_module)

    assert scnn.stream_module.norm.elementwise_affine is elementwise_affine

    stream_output = scnn.forward(image.detach().clone())
    assert len(scnn._last_forward_tiles) > 1
    assert any(y > 0 for y, _, _ in scnn._last_forward_tiles)
    assert any(x > 0 for _, x, _ in scnn._last_forward_tiles)
    torch.testing.assert_close(stream_output, ref_output.detach(), atol=1e-5, rtol=1e-4)

    scnn.backward(image.detach().clone(), upstream_grad.detach().clone())

    stream_module = scnn.stream_module
    reference_grads = {name: param.grad for name, param in reference.named_parameters()}
    streaming_grads = {name: param.grad for name, param in stream_module.named_parameters()}
    assert streaming_grads.keys() == reference_grads.keys()
    for name in reference_grads:
        torch.testing.assert_close(streaming_grads[name], reference_grads[name], atol=1e-5, rtol=1e-4)

    if elementwise_affine:
        assert {"norm.norm.weight", "norm.norm.bias"}.issubset(streaming_grads)
    else:
        assert stream_module.norm.weight is None
        assert stream_module.norm.bias is None
        assert "norm.norm.weight" not in streaming_grads
        assert "norm.norm.bias" not in streaming_grads
        _assert_no_channel_layer_norm_affine_keys(reference)
        _assert_no_channel_layer_norm_affine_keys(stream_module)


def test_scnn_channel_layer_norm_forward_backward_parity():
    _assert_channel_norm_scnn_parity(elementwise_affine=True)


def test_scnn_channel_layer_norm_elementwise_affine_false_forward_backward_parity():
    _assert_channel_norm_scnn_parity(elementwise_affine=False)


def test_streaming_channel_layer_norm_rejects_non_4d_input():
    module = StreamingChannelLayerNorm(3)

    with pytest.raises(ValueError, match="expects 4D NCHW input"):
        module(torch.randn(2, 3, 5))


def test_streaming_channel_layer_norm_rejects_channel_mismatch():
    module = StreamingChannelLayerNorm(3)

    with pytest.raises(ValueError, match="expected 3 channels"):
        module(torch.randn(2, 4, 5, 7))
