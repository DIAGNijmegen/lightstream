import sys
import types

import pytest
import torch

from lightstream.core.scnn import LayerScale, StreamingLayerScale
from lightstream.core.scnn.streaminglayerscale import LayerScale as ImportedLayerScale


def test_layer_scale_broadcasts_supported_shapes():
    x = torch.randn(2, 3, 5, 7)
    for shape in (1, (1,), torch.Size([3, 1, 1]), (1, 3, 1, 1)):
        module = LayerScale(shape, init_value=2.0)
        torch.testing.assert_close(module(x), x * module.scale)


def test_layer_scale_defaults_to_identity_at_initialization_multiplier_zero():
    module = LayerScale((1, 4, 1, 1))
    assert torch.count_nonzero(module.scale) == 0
    torch.testing.assert_close(module(torch.randn(2, 4, 3, 3)), torch.zeros(2, 4, 3, 3))


def test_layer_scale_rejects_unbroadcastable_shape_with_clear_message():
    module = LayerScale((2,))
    with pytest.raises(ValueError, match="cannot broadcast to input shape"):
        module(torch.randn(1, 3, 4, 5))


def test_streaming_layer_scale_round_trip_preserves_state_dict_metadata():
    module = LayerScale((1, 3, 1, 1), init_value=0.25).to(dtype=torch.float64)
    module.scale.requires_grad = False
    module.scale.data.copy_(torch.arange(3, dtype=torch.float64).view(1, 3, 1, 1))

    streaming = StreamingLayerScale.from_layer_scale(module)
    assert list(streaming.state_dict()) == ["scale"]
    assert streaming.scale.dtype == module.scale.dtype
    assert streaming.scale.device == module.scale.device
    assert streaming.scale.requires_grad is False
    torch.testing.assert_close(streaming.scale, module.scale)

    restored = streaming.to_layer_scale()
    assert list(restored.state_dict()) == ["scale"]
    assert restored.scale.dtype == module.scale.dtype
    assert restored.scale.requires_grad is False
    torch.testing.assert_close(restored.scale, module.scale)


def test_streaming_layer_scale_public_export():
    assert ImportedLayerScale is LayerScale
    assert StreamingLayerScale.__name__ == "StreamingLayerScale"


def test_constructor_keeps_layer_scale_streamable(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    from lightstream.core.constructor import StreamingConstructor

    constructor = StreamingConstructor(
        torch.nn.Sequential(LayerScale((1, 3, 1, 1))), tile_size=8, verbose=False
    )
    assert LayerScale in constructor.keep_modules


class SmallLayerScaleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upstream = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.scale = LayerScale((1, 4, 1, 1), init_value=0.5)
        self.downstream = torch.nn.Conv2d(4, 2, kernel_size=3, padding=1)

    def forward(self, x):
        return self.downstream(torch.relu(self.scale(self.upstream(x))))


def test_scnn_layer_scale_forward_backward_parity(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    from lightstream.core.scnn.scnn import StreamingCNN

    torch.manual_seed(404)
    model = SmallLayerScaleNet().eval()
    reference = SmallLayerScaleNet().eval()
    reference.load_state_dict(model.state_dict())

    image = torch.randn(1, 3, 13, 11)
    upstream_grad = torch.randn(1, 2, 13, 11)

    ref_image = image.detach().clone().requires_grad_(True)
    ref_output = reference(ref_image)
    torch.autograd.backward(ref_output, upstream_grad)

    scnn = StreamingCNN(
        model,
        tile_shape=(1, 3, 8, 8),
        verbose=False,
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
    )
    assert isinstance(scnn.stream_module.scale, StreamingLayerScale)

    stream_output = scnn.forward(image.detach().clone())
    torch.testing.assert_close(stream_output, ref_output.detach(), atol=1e-5, rtol=1e-4)

    scnn.backward(image.detach().clone(), upstream_grad.detach().clone())

    reference_grads = {name: param.grad for name, param in reference.named_parameters()}
    streaming_grads = {
        name: param.grad for name, param in scnn.stream_module.named_parameters()
    }
    assert streaming_grads.keys() == reference_grads.keys()
    for name in reference_grads:
        torch.testing.assert_close(
            streaming_grads[name], reference_grads[name], atol=1e-5, rtol=1e-4
        )
