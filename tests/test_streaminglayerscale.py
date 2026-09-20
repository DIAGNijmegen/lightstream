import sys
import types
from copy import deepcopy

import pytest
import torch

from lightstream.core.reducer import MeanReducer, StreamingMeanReducer
from lightstream.core.layers import LayerScale, StreamingLayerScale
from lightstream.core.layers import LayerScale as ImportedLayerScale


def test_layer_scale_scalar_shape_forward_matches_raw_multiplication():
    x = torch.randn(2, 3, 5, 7)
    module = LayerScale(shape=1, init_value=1.75)

    assert tuple(module.weight.shape) == (1,)
    torch.testing.assert_close(module(x), x * module.weight)


def test_layer_scale_channel_shape_broadcasts_across_spatial_dimensions():
    x = torch.randn(2, 3, 5, 7)
    module = LayerScale(shape=(1, 3, 1, 1), init_value=1.0)
    module.weight.data.copy_(torch.tensor([[[[0.5]], [[1.5]], [[2.5]]]]))

    expected = x * torch.tensor([0.5, 1.5, 2.5]).view(1, 3, 1, 1)
    torch.testing.assert_close(module(x), expected)


def test_layer_scale_broadcasts_supported_shapes():
    x = torch.randn(2, 3, 5, 7)
    for shape in (1, (1,), torch.Size([3, 1, 1]), (1, 3, 1, 1)):
        module = LayerScale(shape, init_value=2.0)
        torch.testing.assert_close(module(x), x * module.weight)


def test_layer_scale_defaults_to_identity_at_initialization_multiplier_zero():
    module = LayerScale((1, 4, 1, 1))
    assert torch.count_nonzero(module.weight) == 0
    torch.testing.assert_close(module(torch.randn(2, 4, 3, 3)), torch.zeros(2, 4, 3, 3))


def test_layer_scale_rejects_unbroadcastable_shape_with_clear_message():
    module = LayerScale((2,))
    with pytest.raises(ValueError, match="cannot broadcast to input shape"):
        module(torch.randn(1, 3, 4, 5))


def test_streaming_layer_scale_round_trip_preserves_state_dict_metadata():
    module = LayerScale((1, 3, 1, 1), init_value=0.25).to(dtype=torch.float64)
    module.weight.requires_grad = False
    module.weight.data.copy_(torch.arange(3, dtype=torch.float64).view(1, 3, 1, 1))

    streaming = StreamingLayerScale.from_layer_scale(module)
    assert list(streaming.state_dict()) == ["weight"]
    assert streaming.weight.dtype == module.weight.dtype
    assert streaming.weight.device == module.weight.device
    assert streaming.weight.requires_grad is False
    torch.testing.assert_close(streaming.weight, module.weight)

    restored = streaming.to_layer_scale()
    assert list(restored.state_dict()) == ["weight"]
    assert restored.weight.dtype == module.weight.dtype
    assert restored.weight.requires_grad is False
    torch.testing.assert_close(restored.weight, module.weight)


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


class SmallLayerScaleReducerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            LayerScale((1, 4, 1, 1), init_value=0.5),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(4, 2, kernel_size=1, bias=False),
            MeanReducer(),
        )

    def forward(self, x):
        return self.head(self.features(x))


def test_scnn_layer_scale_reducer_head_forward_backward_scale_gradient_parity():
    from lightstream.core.scnn.scnn import StreamingCNN

    torch.manual_seed(405)
    model = SmallLayerScaleReducerNet().eval()
    reference = SmallLayerScaleReducerNet().eval()
    reference.load_state_dict(model.state_dict())

    image = torch.randn(1, 3, 13, 11)

    ref_image = image.detach().clone().requires_grad_(True)
    ref_output = reference(ref_image)
    upstream_grad = torch.randn_like(ref_output)
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
    assert isinstance(scnn.stream_module.features[1], StreamingLayerScale)
    assert isinstance(scnn.stream_module.head[1], StreamingMeanReducer)

    stream_output = scnn.forward(image.detach().clone())
    torch.testing.assert_close(stream_output, ref_output.detach(), atol=1e-5, rtol=1e-4)

    scnn.backward(image.detach().clone(), upstream_grad.detach().clone())

    torch.testing.assert_close(
        scnn.stream_module.features[1].weight.grad,
        reference.features[1].weight.grad,
        atol=1e-5,
        rtol=1e-4,
    )


class SpatiallyDependentLayerScaleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upstream = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.scale = LayerScale((1, 4, 1, 1), init_value=0.625)
        self.downstream = torch.nn.Conv2d(4, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.downstream(torch.relu(self.scale(self.upstream(x))))


def test_scnn_layer_scale_accumulates_spatial_dependencies_across_shifted_tiles(
    monkeypatch,
):
    """A pre-spatial scale must retain dependency gradients from tile halos."""
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    from lightstream.core.scnn.scnn import StreamingCNN

    torch.manual_seed(406)
    model = SpatiallyDependentLayerScaleNet().eval()
    reference = SpatiallyDependentLayerScaleNet().eval()
    reference.load_state_dict(model.state_dict())

    image_shape = (13, 14)
    tile_shape = (8, 9)
    scnn = StreamingCNN(
        model,
        tile_shape=(1, 3, *tile_shape),
        verbose=False,
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
    )

    replay_records = []

    def record_forward(module, inputs, _output):
        replay_records.append(
            {
                "input": inputs[0].detach().clone(),
                "input_loc": deepcopy(module.input_loc),
                "grad_lost": deepcopy(module.grad_lost),
                "ownership": deepcopy(module.input_loc.sides),
            }
        )

    def record_backward(_module, _grad_input, grad_output):
        replay_records[-1]["grad_output"] = grad_output[0].detach().clone()

    scnn.stream_module.scale.register_forward_hook(record_forward)
    scnn.stream_module.scale.register_full_backward_hook(record_backward)

    for cycle in range(2):
        cycle_record_start = len(replay_records)
        reference.zero_grad(set_to_none=True)
        scnn.stream_module.zero_grad(set_to_none=True)
        torch.manual_seed(407 + cycle)
        image = torch.randn(1, 3, *image_shape)
        upstream = torch.randn(1, 3, *image_shape)

        reference_output = reference(image)
        reference_output.backward(upstream)
        streaming_output = scnn(image)
        torch.testing.assert_close(streaming_output, reference_output.detach())

        tile_starts = [(y, x) for y, x, _ in scnn._last_forward_tiles]
        assert any(
            0 < next_x - x < tile_shape[1]
            for (y, x), (next_y, next_x) in zip(tile_starts, tile_starts[1:])
            if y == next_y
        )
        # The last replay row is shifted back from the regular grid and
        # overlaps the preceding row in both its input and dependency halo.
        assert tile_starts[-1] == (5, 5)
        scnn.backward(image.detach().clone(), upstream)

        cycle_records = replay_records[cycle_record_start + len(tile_starts) :]
        assert len(cycle_records) == len(tile_starts)
        assert all(
            set(record) == {
                "input",
                "grad_output",
                "input_loc",
                "grad_lost",
                "ownership",
            }
            for record in cycle_records
        )
        complete_tile_gradient = sum(
            (record["grad_output"] * record["input"]).sum_to_size(
                scnn.stream_module.scale.weight.shape
            )
            for record in cycle_records
        )
        torch.testing.assert_close(
            complete_tile_gradient,
            scnn.stream_module.scale.weight.grad,
            atol=1e-5,
            rtol=1e-4,
        )

        for name in (
            "scale.weight",
            "upstream.weight",
            "upstream.bias",
            "downstream.weight",
            "downstream.bias",
        ):
            torch.testing.assert_close(
                dict(scnn.stream_module.named_parameters())[name].grad,
                dict(reference.named_parameters())[name].grad,
                atol=1e-5,
                rtol=1e-4,
                msg=f"cycle {cycle}: {name}",
            )
