import torch
import torch.nn as nn
import pytest

from lightstream.core.constructor import StreamingConstructor

from lightstream.core.reducer import (
    BaseReducer,
    BaseStreamingGlobalReducer,
    GeMReducer,
    MeanReducer,
    StreamingGeMReducer,
    StreamingMeanReducer,
    StreamingSumReducer,
    SumReducer,
)


class AllReducerHeadsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.sum_head = nn.Sequential(nn.Conv2d(6, 2, kernel_size=1, bias=False), SumReducer())
        self.mean_head = nn.Sequential(nn.Conv2d(6, 3, kernel_size=1, bias=False), MeanReducer())

    def forward(self, x):
        feat = self.backbone(x)
        return self.sum_head(feat), self.mean_head(feat)


class DownsampledReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.reducer = MeanReducer()

    def forward(self, x, mask: torch.Tensor | None = None):
        feat = self.down(x)
        return self.reducer(feat, mask=mask)


class MixedHeadsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.raw_head = nn.Conv2d(5, 4, kernel_size=1, bias=False)
        self.reducer_head = nn.Sequential(nn.Conv2d(5, 4, kernel_size=1, bias=False), MeanReducer())

    def forward(self, x):
        feat = self.backbone(x)
        return {"raw": self.raw_head(feat), "reduced": self.reducer_head(feat)}


class GeMHeadNet(nn.Module):
    def __init__(self, learnable_r: bool):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.gem_head = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=1, bias=False),
            GeMReducer(r=3.2, eps=1e-6, learnable_r=learnable_r),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.gem_head(feat)


class ValueLogitsReducer(BaseReducer):
    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError(f"ValueLogitsReducer expects exactly two inputs, got {len(inputs)}")
        value, logits = inputs
        if value.ndim != 4 or logits.ndim != 4:
            raise ValueError("ValueLogitsReducer expects NCHW inputs")
        if value.shape != logits.shape:
            raise ValueError(f"ValueLogitsReducer shape mismatch value={tuple(value.shape)} logits={tuple(logits.shape)}")
        if self._streaming_passthrough:
            self._last_inputs = inputs
            self._last_output = value
            return value
        weights = torch.softmax(logits, dim=-1)
        if mask is not None:
            mask = mask.to(device=value.device, dtype=value.dtype)[None, None]
            weights = weights * mask
        return (value * weights).sum(dim=(-2, -1), keepdim=True)

    def to_streaming(self) -> BaseStreamingGlobalReducer:
        return StreamingValueLogitsReducer()


class StreamingValueLogitsReducer(BaseStreamingGlobalReducer):
    def __init__(self):
        super().__init__(mode="sum")

    def init_reduction_state(self, *, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, accumulator_dtype: torch.dtype) -> None:
        self.running_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=dtype)

    def accumulate_valid_tile(self, tile: torch.Tensor | tuple[torch.Tensor, ...], valid_mask: torch.Tensor) -> None:
        value, logits = self._parse_multi_input_payload(tile)
        weights = torch.softmax(logits, dim=-1)
        mask4 = valid_mask.to(device=value.device, dtype=value.dtype)[None, None]
        self.running_sum = self.running_sum + (value * weights * mask4).sum(dim=(-2, -1), keepdim=True)

    def finalize_from_state(self) -> torch.Tensor:
        return self.running_sum

    def reduce_tile_for_backward(self, trimmed_output: torch.Tensor | tuple[torch.Tensor, ...], valid_mask: torch.Tensor | None, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        value, logits = self._parse_multi_input_payload(trimmed_output)
        weights = torch.softmax(logits, dim=-1)
        if valid_mask is not None:
            mask4 = valid_mask.to(device=value.device, dtype=value.dtype)[None, None]
            return (value * weights * mask4).sum(dim=(-2, -1), keepdim=True)
        return (value * weights).sum(dim=(-2, -1), keepdim=True)


class ValueLogitsHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False), nn.ReLU())
        self.value_head = nn.Conv2d(6, 3, kernel_size=1, bias=False)
        self.logit_head = nn.Conv2d(6, 3, kernel_size=1, bias=False)
        self.reducer = ValueLogitsReducer()

    def forward(self, x, mask: torch.Tensor | None = None):
        feat = self.backbone(x)
        value = self.value_head(feat)
        logits = self.logit_head(feat)
        reduced = self.reducer(value, logits, mask=mask)
        return reduced, value, logits


def _make_streaming(model: nn.Module, tile_size: int = 4):
    constructor = StreamingConstructor(
        model,
        tile_size=tile_size,
        verbose=False,
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=True,
        normalize_on_gpu=False,
    )
    return constructor.prepare_streaming_model()


def test_scnn_forward_all_reducer_heads_parity():
    torch.manual_seed(7)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 9, 11)

    with torch.no_grad():
        expected_sum, expected_mean = model(image)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image)

    assert streamed_sum.shape == expected_sum.shape
    assert streamed_mean.shape == expected_mean.shape
    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_scnn_forward_mixed_reducer_non_reducer_heads_parity():
    torch.manual_seed(11)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 10, 12)

    with torch.no_grad():
        expected = model(image)

    scnn = _make_streaming(model, tile_size=5)
    with torch.no_grad():
        streamed = scnn.forward(image)

    assert streamed["raw"].shape == expected["raw"].shape
    assert streamed["reduced"].shape == expected["reduced"].shape
    assert torch.allclose(streamed["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)


def test_scnn_forward_border_tiles_parity_small_image_large_tile():
    torch.manual_seed(3)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 3, 4)

    with torch.no_grad():
        expected = model(image)

    scnn = _make_streaming(model, tile_size=8)
    with torch.no_grad():
        streamed = scnn.forward(image)

    assert torch.allclose(streamed["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)


def test_scnn_backward_reducer_replay_consumed_debug_mode():
    torch.manual_seed(23)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 9, 11)

    scnn = _make_streaming(model, tile_size=4)
    scnn.debug_reducer_replay = True

    with torch.no_grad():
        streamed_output = scnn.forward(image)

    grad = tuple(torch.ones_like(head) for head in streamed_output)
    # no exception means replay assignments were consumed and validated
    scnn.backward(image, grad)


def test_scnn_backward_structure_mismatch_error():
    torch.manual_seed(29)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 6, 7)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed = scnn.forward(image)

    wrong_grad = (torch.ones_like(streamed["raw"]), torch.ones_like(streamed["reduced"]))
    with pytest.raises(ValueError, match="Gradient output structure does not match"):
        scnn.backward(image, wrong_grad)


def test_scnn_backward_reducer_head_gradient_shape_mismatch_error():
    torch.manual_seed(31)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 8, 9)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed = scnn.forward(image)

    wrong_grad = {
        "raw": torch.ones_like(streamed["raw"]),
        "reduced": torch.ones((streamed["reduced"].shape[0], streamed["reduced"].shape[1], 2, 1)),
    }
    with pytest.raises(ValueError, match="Reducer-backed head expects gradient"):
        scnn.backward(image, wrong_grad)


def test_scnn_backward_requires_forward_for_reducer_heads():
    torch.manual_seed(41)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 9, 11)

    scnn = _make_streaming(model, tile_size=4)

    with torch.no_grad():
        expected_output = model(image)

    grad = tuple(torch.ones_like(head) for head in expected_output)
    with pytest.raises(RuntimeError, match="requires prior streaming forward pass"):
        scnn.backward(image, grad)


@pytest.mark.parametrize(
    ("head_module_name", "expected_streaming_type"),
    [
        ("sum_head.1", StreamingSumReducer),
        ("mean_head.1", StreamingMeanReducer),
    ],
)
def test_scnn_converts_reducer_heads_to_streaming_types(head_module_name, expected_streaming_type):
    model = AllReducerHeadsNet().eval()
    scnn = _make_streaming(model, tile_size=4)
    assert isinstance(scnn.stream_module.get_submodule(head_module_name), expected_streaming_type)


@pytest.mark.parametrize("learnable_r", [False, True])
def test_scnn_gem_conversion_forward_backward_parity(learnable_r):
    torch.manual_seed(53 if learnable_r else 47)
    model = GeMHeadNet(learnable_r=learnable_r).eval()
    reference = GeMHeadNet(learnable_r=learnable_r).eval()
    reference.load_state_dict(model.state_dict())

    image = (torch.rand(1, 3, 9, 11) + 0.05).requires_grad_(True)
    ref_image = image.detach().clone().requires_grad_(True)

    ref_out = reference(ref_image)
    ref_grad = torch.full_like(ref_out, 0.37)
    torch.autograd.backward(ref_out, ref_grad)

    scnn = _make_streaming(model, tile_size=4)
    assert isinstance(scnn.stream_module.get_submodule("gem_head.1"), StreamingGeMReducer)
    stream_out = scnn.forward(image.detach().clone())
    assert torch.allclose(stream_out, ref_out.detach(), atol=1e-5, rtol=1e-4)

    scnn.backward(image.detach().clone(), ref_grad.detach().clone())

    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}
    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    for name, ref_param_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_param_grad, atol=1e-5, rtol=1e-4), name

    # input-grad parity under real SCNN execution
    input_scnn = _make_streaming(GeMHeadNet(learnable_r=learnable_r).eval(), tile_size=4)
    input_scnn.gather_input_gradient = True
    input_scnn._remove_hooks()
    input_scnn._add_hooks_for_streaming()
    scnn_image = image.detach().clone()
    _ = input_scnn.forward(scnn_image)
    input_scnn.backward(scnn_image, ref_grad.detach().clone())
    assert torch.allclose(input_scnn.saliency_map, ref_image.grad, atol=1e-5, rtol=1e-4)

    if learnable_r:
        assert reference.gem_head[1].r.grad is not None
        assert stream_grads["gem_head.1.r"] is not None
        assert torch.allclose(stream_grads["gem_head.1.r"], reference.gem_head[1].r.grad, atol=1e-5, rtol=1e-4)


def test_scnn_mixed_head_reducer_mapping_stable():
    torch.manual_seed(61)
    model = MixedHeadsNet().eval()
    image = torch.randn(1, 3, 13, 9)

    scnn = _make_streaming(model, tile_size=4)
    assert isinstance(scnn.stream_module.reducer_head[1], StreamingMeanReducer)
    with torch.no_grad():
        expected = model(image)
        stream_first = scnn.forward(image)
        stream_second = scnn.forward(image)

    assert scnn._reducer_head_map
    reducer_head_index = next(iter(scnn._reducer_head_map.keys()))
    assert reducer_head_index == 1
    assert torch.allclose(stream_first["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(stream_first["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(stream_second["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(stream_second["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)


def test_scnn_multi_input_reducer_forward_odd_borders_with_mask_parity():
    torch.manual_seed(101)
    model = ValueLogitsHeadNet().eval()
    image = torch.randn(1, 3, 9, 13)
    mask = (torch.arange(9)[:, None] + torch.arange(13)[None, :]) % 3 != 0

    with torch.no_grad():
        expected, _, _ = model(image, mask=mask)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed, _, _ = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed, expected, atol=1e-5, rtol=1e-4)


def test_scnn_multi_input_reducer_backward_positional_input_parity_streaming_vs_nonstreaming():
    torch.manual_seed(103)
    model = ValueLogitsHeadNet().eval()
    reference = ValueLogitsHeadNet().eval()
    reference.load_state_dict(model.state_dict())
    image = torch.randn(1, 3, 11, 9)
    mask = (torch.rand(11, 9) > 0.25)

    ref_image = image.clone().requires_grad_(True)
    ref_reduced, ref_value, ref_logits = reference(ref_image, mask=mask)
    grad_reduced = torch.full_like(ref_reduced, 0.21)
    grad_value = torch.zeros_like(ref_value)
    grad_logits = torch.zeros_like(ref_logits)
    torch.autograd.backward((ref_reduced, ref_value, ref_logits), (grad_reduced, grad_value, grad_logits))
    ref_grads = {name: p.grad.detach().clone() for name, p in reference.named_parameters() if p.grad is not None}

    scnn = _make_streaming(model, tile_size=5)
    stream_reduced, stream_value, stream_logits = scnn.forward(image.clone(), mask=mask)
    assert torch.allclose(stream_reduced, ref_reduced.detach(), atol=1e-5, rtol=1e-4)
    scnn.backward(image.clone(), (grad_reduced, grad_value, grad_logits), mask=mask)

    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}
    for name, ref_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_grad, atol=1e-5, rtol=1e-4), name


def test_scnn_single_input_reducers_compatibility_unchanged():
    torch.manual_seed(107)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 7, 15)
    mask = (torch.rand(7, 15) > 0.4)

    with torch.no_grad():
        feat = model.backbone(image)
        expected_sum = model.sum_head[1](model.sum_head[0](feat), mask=mask)
        expected_mean = model.mean_head[1](model.mean_head[0](feat), mask=mask)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_scnn_input_resolution_mask_for_input_resolution_reducer():
    torch.manual_seed(115)
    model = AllReducerHeadsNet().eval()
    image = torch.randn(1, 3, 8, 10)
    mask = ((torch.arange(8)[:, None] + 2 * torch.arange(10)[None, :]) % 4 != 0).to(torch.float32)

    with torch.no_grad():
        feat = model.backbone(image)
        expected_sum = model.sum_head[1](model.sum_head[0](feat), mask=mask.to(torch.bool))
        expected_mean = model.mean_head[1](model.mean_head[0](feat), mask=mask.to(torch.bool))

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_scnn_downsampled_reducer_mask_matches_reduced_feature_map():
    torch.manual_seed(117)
    model = DownsampledReducerNet().eval()
    image = torch.randn(1, 3, 9, 11)
    mask = (torch.arange(5)[:, None] + torch.arange(6)[None, :]) % 2 == 0

    with torch.no_grad():
        expected = model(image, mask=mask)

    scnn = _make_streaming(model, tile_size=5)
    with torch.no_grad():
        streamed = scnn.forward(image, mask=mask)

    assert torch.allclose(streamed, expected, atol=1e-5, rtol=1e-4)


def test_scnn_too_small_reducer_mask_fails_at_reducer_slice_site():
    torch.manual_seed(119)
    model = DownsampledReducerNet().eval()
    image = torch.randn(1, 3, 9, 11)
    mask = torch.ones((4, 6), dtype=torch.bool)

    scnn = _make_streaming(model, tile_size=5)
    with pytest.raises(ValueError, match="reducer/reduced feature spatial domain"):
        with torch.no_grad():
            scnn.forward(image, mask=mask)


def test_scnn_multi_input_reducer_failure_on_shape_mismatch():
    torch.manual_seed(109)
    model = ValueLogitsHeadNet().eval()
    image = torch.randn(1, 3, 9, 9)
    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        _ = scnn.forward(image)

    scnn._reducer_input_indices = {0: (0, 1)}
    bad_outputs = (torch.randn(1, 3, 3, 3), torch.randn(1, 3, 2, 3), torch.randn(1, 3, 3, 3))
    with pytest.raises(RuntimeError, match="tile input spatial mismatch"):
        scnn._stitch_forward_outputs([None, None, None], bad_outputs, 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), None)


def test_scnn_multi_input_reducer_failure_on_input_order_mismatch():
    torch.manual_seed(113)
    model = ValueLogitsHeadNet().eval()
    scnn = _make_streaming(model, tile_size=4)
    scnn._reducer_head_map = {0: scnn.stream_module.reducer}
    scnn._reducer_input_indices = {0: (0, 2, 1)}
    with pytest.raises(RuntimeError, match="input index order mismatch"):
        scnn._stitch_forward_outputs([None, None, None], (torch.randn(1, 3, 3, 3),) * 3, 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), None)