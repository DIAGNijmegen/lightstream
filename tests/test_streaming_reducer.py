import copy

import torch
import torch.nn as nn
import pytest

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.models.segment.streamingwss import StreamingWSS

from lightstream.core.reducer import (
    AttentionGeMReducer,
    GeMReducer,
    MeanReducer,
    StreamingAttentionGeMReducer,
    StreamingMeanReducer,
    StreamingSumReducer,
    SumReducer,
)


class MixedReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.raw_head = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.mean_head = nn.Sequential(nn.Conv2d(4, 2, kernel_size=1, bias=False), MeanReducer())

    def forward(self, x):
        feat = self.backbone(x)
        return {"raw": self.raw_head(feat), "reduced": self.mean_head(feat)}


class MultiReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.sum_head = nn.Sequential(nn.Conv2d(5, 2, kernel_size=1, bias=False), SumReducer())
        self.mean_head = nn.Sequential(nn.Conv2d(5, 2, kernel_size=1, bias=False), MeanReducer())

    def forward(self, x):
        feat = self.backbone(x)
        return self.sum_head(feat), self.mean_head(feat)


def _require_and_retain_output_grads(outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    for output in outputs:
        output.requires_grad_(True)
        output.retain_grad()
    return outputs


def _assert_grad_matches(
    stream_grads: dict[str, torch.Tensor],
    ref_grads: dict[str, torch.Tensor],
    name: str,
    *,
    atol: float = 3e-4,
    rtol: float = 3e-3,
) -> None:
    assert name in stream_grads, name
    assert name in ref_grads, name
    assert torch.isfinite(stream_grads[name]).all(), name
    assert torch.isfinite(ref_grads[name]).all(), name
    assert torch.allclose(stream_grads[name], ref_grads[name], atol=atol, rtol=rtol), name


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


def test_streaming_reducer_mixed_outputs_forward_parity():
    torch.manual_seed(1)
    model = MixedReducerNet().eval()
    image = torch.randn(1, 3, 9, 11)

    with torch.no_grad():
        expected = model(image)

    scnn = _make_streaming(model, tile_size=4)
    with torch.no_grad():
        streamed = scnn.forward(image)

    assert streamed["raw"].shape == expected["raw"].shape
    assert streamed["reduced"].shape == expected["reduced"].shape
    assert torch.allclose(streamed["raw"], expected["raw"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed["reduced"], expected["reduced"], atol=1e-5, rtol=1e-4)


def test_streaming_reducer_multiple_reducers_modes_forward_parity():
    torch.manual_seed(2)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 8, 10)

    with torch.no_grad():
        expected_sum, expected_mean = model(image)

    scnn = _make_streaming(model, tile_size=5)
    with torch.no_grad():
        streamed_sum, streamed_mean = scnn.forward(image)

    assert streamed_sum.shape == expected_sum.shape
    assert streamed_mean.shape == expected_mean.shape
    assert torch.allclose(streamed_sum, expected_sum, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed_mean, expected_mean, atol=1e-5, rtol=1e-4)


def test_streaming_reducer_backward_parity():
    torch.manual_seed(3)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 11, 9)

    reference = MultiReducerNet().eval()
    reference.load_state_dict(model.state_dict())

    ref_image = image.clone().requires_grad_(True)
    ref_sum, ref_mean = reference(ref_image)
    ref_loss = (0.7 * ref_sum).sum() + (1.3 * ref_mean).sum()
    ref_loss.backward()

    scnn = _make_streaming(model, tile_size=4)
    streamed_sum, streamed_mean = scnn.forward(image.clone())
    grad_sum = torch.full_like(streamed_sum, 0.7)
    grad_mean = torch.full_like(streamed_mean, 1.3)
    scnn.backward(image.clone(), (grad_sum, grad_mean))

    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}

    for name, ref_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_grad, atol=1e-5, rtol=1e-4), name


def test_streaming_reducer_backward_parity_tiny_odd_image():
    torch.manual_seed(5)
    model = MultiReducerNet().eval()
    image = torch.randn(1, 3, 3, 5)

    reference = MultiReducerNet().eval()
    reference.load_state_dict(model.state_dict())

    ref_image = image.clone().requires_grad_(True)
    ref_sum, ref_mean = reference(ref_image)
    ref_loss = (0.2 * ref_sum).sum() + (-0.4 * ref_mean).sum()
    ref_loss.backward()

    scnn = _make_streaming(model, tile_size=6)
    streamed_sum, streamed_mean = scnn.forward(image.clone())
    grad_sum = torch.full_like(streamed_sum, 0.2)
    grad_mean = torch.full_like(streamed_mean, -0.4)
    scnn.backward(image.clone(), (grad_sum, grad_mean))

    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}

    for name, ref_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_grad, atol=1e-5, rtol=1e-4), name


def test_streaming_mean_reducer_running_count_uses_fp32_accumulator():
    reducer = StreamingMeanReducer()
    tile = torch.ones((1, 2, 2, 3), dtype=torch.float16)

    reducer.accumulate_valid_tile(tile, valid_mask=torch.ones((2, 3), dtype=torch.bool))

    assert reducer.running_sum.dtype == torch.float16
    assert reducer.running_count.dtype == torch.float32

    output = reducer.finalize_stream()
    assert output.dtype == tile.dtype
    assert torch.allclose(output, torch.ones((1, 2, 1, 1), dtype=tile.dtype))


def test_streaming_sum_reducer_does_not_normalize():
    reducer = StreamingSumReducer()
    tile = torch.ones((1, 2, 2, 3), dtype=torch.float16)

    reducer.accumulate_valid_tile(tile, valid_mask=torch.ones((2, 3), dtype=torch.bool))

    output = reducer.finalize_stream()
    assert output.dtype == tile.dtype
    assert torch.allclose(output, torch.full((1, 2, 1, 1), 6.0, dtype=tile.dtype))


def test_single_input_reducer_contract_remains_unchanged():
    reducer = MeanReducer()
    x = torch.randn(1, 2, 3, 5)
    y = reducer(x)
    assert torch.allclose(y, x.mean(dim=(-2, -1), keepdim=True))


def test_single_input_reducer_rejects_multi_input_arity():
    reducer = MeanReducer()
    x = torch.randn(1, 2, 3, 5)
    with pytest.raises(ValueError, match="expects exactly one tensor input"):
        reducer(x, x)

from lightstream.core.reducer import StreamingGeMReducer


def _gem_reference(x: torch.Tensor, mask: torch.Tensor, r: torch.Tensor, eps: float):
    x_clamped = x.clamp_min(eps)
    mask4 = mask.to(dtype=x.dtype, device=x.device)[None, None]
    n = mask4.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    m = (x_clamped.pow(r) * mask4).sum(dim=(-2, -1), keepdim=True) / n
    return m.clamp_min(eps).pow(1.0 / r)


def test_streaming_gem_default_r_init():
    reducer = StreamingGeMReducer()
    assert torch.isclose(reducer.current_r, torch.tensor(4.0), atol=1e-6)


def test_streaming_gem_r_init_matches_init():
    reducer = StreamingGeMReducer(r_init=16.0)
    assert torch.isclose(reducer.current_r, torch.tensor(16.0), atol=1e-6)


def test_streaming_gem_forward_parity_masked_tiny_odd_shape():
    torch.manual_seed(7)
    reducer = StreamingGeMReducer(r_init=3.5, eps=1e-6)
    x = torch.rand(1, 2, 3, 5) + 0.01
    mask = torch.tensor([[1, 0, 1, 0, 1], [1, 1, 0, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.bool)

    reducer.start_stream(output_height=3, output_width=5, batch_size=1, channels=2, device=x.device, dtype=x.dtype)
    reducer.accumulate_stream_tile(x[:, :, :2, :3], 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,2,0,3), user_mask=mask[:2,:3])
    reducer.accumulate_stream_tile(x[:, :, :2, 3:], 0, 1, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,2,3,5), user_mask=mask[:2,3:])
    reducer.accumulate_stream_tile(x[:, :, 2:, :], 1, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (2,3,0,5), user_mask=mask[2:,:])
    y_stream = reducer.finish_stream()

    y_ref = _gem_reference(x, mask, reducer.current_r.to(dtype=x.dtype), reducer.eps)
    assert torch.allclose(y_stream, y_ref, atol=1e-5, rtol=1e-4)


def test_streaming_gem_backward_input_grad_parity():
    torch.manual_seed(9)
    x = (torch.rand(1, 3, 5, 7) + 0.05).requires_grad_(True)
    mask = (torch.rand(5, 7) > 0.3)

    reducer = StreamingGeMReducer(r_init=2.3, eps=1e-6)
    reducer.start_stream(output_height=5, output_width=7, batch_size=1, channels=3, device=x.device, dtype=x.dtype)
    reducer.accumulate_stream_tile(x[:, :, :, :4], 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,5,0,4), user_mask=mask[:, :4])
    reducer.accumulate_stream_tile(x[:, :, :, 4:], 0, 1, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,5,4,7), user_mask=mask[:, 4:])
    y_stream = reducer.finish_stream()
    loss_stream = y_stream.sum()
    loss_stream.backward()
    grad_x_stream = x.grad.detach().clone()
    x_ref = x.detach().clone().requires_grad_(True)
    r_ref = reducer.r.detach().clone()
    y_ref = _gem_reference(x_ref, mask, r_ref.to(dtype=x_ref.dtype), reducer.eps)
    y_ref.sum().backward()

    assert torch.allclose(grad_x_stream, x_ref.grad, atol=1e-5, rtol=1e-4)


def test_streaming_gem_backward_replay_surrogate_matches_global_gradient():
    torch.manual_seed(31)
    x = (torch.rand(1, 2, 4, 6) + 0.05).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    mask = torch.tensor(
        [
            [1, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    upstream = torch.tensor([[[[0.17]], [[-0.29]]]], dtype=x.dtype)
    sides = type("S", (), dict(top=False, left=False, right=False, bottom=False))()

    reducer = StreamingGeMReducer(r_init=3.2, eps=1e-6)
    reducer.start_stream(output_height=4, output_width=6, batch_size=1, channels=2, device=x.device, dtype=x.dtype)
    reducer.accumulate_stream_tile(x[:, :, :, :3], 0, 0, sides, (0, 4, 0, 3), user_mask=mask[:, :3])
    reducer.accumulate_stream_tile(x[:, :, :, 3:], 0, 1, sides, (0, 4, 3, 6), user_mask=mask[:, 3:])

    global_context = reducer.extra_state_for_backward()
    replay_left = reducer.reduce_tile_for_backward(x[:, :, :, :3], mask[:, :3], global_context)
    replay_right = reducer.reduce_tile_for_backward(x[:, :, :, 3:], mask[:, 3:], global_context)
    torch.autograd.backward((replay_left, replay_right), (upstream, upstream))

    y_ref = _gem_reference(x_ref, mask, reducer.current_r.to(dtype=x_ref.dtype), reducer.eps)
    torch.autograd.backward(y_ref, upstream)

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-4)


def test_streaming_gem_reducer_backward_parity_with_non_streaming_gradients():
    torch.manual_seed(37)
    model = GeMHeadNet().eval()
    reference = GeMHeadNet().eval()
    reference.load_state_dict(model.state_dict())

    image = torch.rand(1, 3, 9, 11) + 0.05
    ref_image = image.detach().clone().requires_grad_(True)
    grad_out = torch.tensor([[[[0.17]], [[-0.29]], [[0.43]]]], dtype=image.dtype)

    ref_out = reference(ref_image)
    torch.autograd.backward(ref_out, grad_out)

    scnn = _make_streaming(model, tile_size=4)
    scnn.debug_reducer_replay = True
    stream_out = scnn.forward(image.detach().clone())
    assert torch.allclose(stream_out, ref_out.detach(), atol=1e-5, rtol=1e-4)
    scnn.backward(image.detach().clone(), grad_out.detach().clone())

    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}
    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}

    for name in ("backbone.0.weight", "head.weight"):
        assert name in stream_grads
        assert name in ref_grads
        assert torch.allclose(stream_grads[name], ref_grads[name], atol=2e-5, rtol=2e-4), name


def test_streaming_gem_fp16_stability_accumulator_fp32():
    torch.manual_seed(11)
    reducer = StreamingGeMReducer(r_init=4.0)
    x = (torch.rand(1, 2, 4, 4, dtype=torch.float16) + 0.01)
    mask = torch.ones((4, 4), dtype=torch.bool)

    reducer.start_stream(output_height=4, output_width=4, batch_size=1, channels=2, device=x.device, dtype=x.dtype)
    reducer.accumulate_stream_tile(x, 0, 0, type('S', (), dict(top=False,left=False,right=False,bottom=False))(), (0,4,0,4), user_mask=mask)
    y = reducer.finish_stream()

    assert reducer.running_count.dtype == torch.float32
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()


class GeMHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        self.reducer = GeMReducer(r_init=2.7, eps=1e-6)

    def forward(self, x):
        feat = self.backbone(x)
        return self.reducer(self.head(feat))


class AttentionGeMHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False), nn.ReLU())
        self.feat_head = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        self.logit_head = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.reducer = AttentionGeMReducer(r_init=2.7, eps=1e-6)

    def forward(self, x):
        feat = self.backbone(x)
        return self.reducer(self.feat_head(feat), self.logit_head(feat))


class AttentionGeMBiasHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False), nn.ReLU())
        self.feat_head = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        self.logit_head = nn.Conv2d(4, 1, kernel_size=1, bias=True)
        self.reducer = AttentionGeMReducer(r_init=2.7, eps=1e-6)

    def forward(self, x):
        feat = self.backbone(x)
        return self.reducer(self.feat_head(feat), self.logit_head(feat))


def test_streaming_attention_gem_uniform_bias_backward_replay_matches_non_streaming():
    torch.manual_seed(41)
    x = torch.tensor(
        [
            [
                [
                    [0.11, 0.23, 0.37, 0.53],
                    [0.71, 0.89, 1.07, 1.31],
                    [1.47, 1.61, 1.79, 1.97],
                ],
                [
                    [0.29, 0.43, 0.59, 0.73],
                    [0.97, 1.13, 1.29, 1.43],
                    [1.67, 1.83, 2.03, 2.19],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    upstream = torch.tensor([[[[0.17]], [[-0.31]]]], dtype=x.dtype)
    sides = type("S", (), dict(top=False, left=False, right=False, bottom=False))()

    ref_bias = nn.Parameter(torch.tensor(0.37, dtype=x.dtype))
    ref_reducer = AttentionGeMReducer(r_init=2.5, eps=1e-6)
    ref_logits = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype) + ref_bias
    ref_out = ref_reducer(x, ref_logits)
    torch.autograd.backward(ref_out, upstream)

    stream_bias = nn.Parameter(ref_bias.detach().clone())
    stream_reducer = StreamingAttentionGeMReducer(r_init=2.5, eps=1e-6)
    stream_reducer.start_stream(
        output_height=x.shape[-2],
        output_width=x.shape[-1],
        batch_size=x.shape[0],
        channels=x.shape[1],
        device=x.device,
        dtype=x.dtype,
    )

    with torch.no_grad():
        left_x = x[:, :, :, :3]
        right_x = x[:, :, :, 1:]
        left_logits = torch.zeros((x.shape[0], 1, x.shape[2], 3), dtype=x.dtype) + stream_bias
        right_logits = torch.zeros((x.shape[0], 1, x.shape[2], 3), dtype=x.dtype) + stream_bias
        stream_reducer.accumulate_stream_tile((left_x, left_logits), 0, 0, sides, (0, 3, 0, 3))
        stream_reducer.accumulate_stream_tile((right_x, right_logits), 0, 1, sides, (0, 3, 1, 4))
        stream_out = stream_reducer.finish_stream()

    assert torch.allclose(stream_out, ref_out.detach(), atol=1e-6, rtol=1e-5)

    left_logits = torch.zeros((x.shape[0], 1, x.shape[2], 3), dtype=x.dtype) + stream_bias
    right_logits = torch.zeros((x.shape[0], 1, x.shape[2], 3), dtype=x.dtype) + stream_bias
    left_valid = torch.ones((x.shape[2], 3), dtype=torch.bool)
    right_valid = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.bool)
    global_context = stream_reducer.extra_state_for_backward()
    replay_left = stream_reducer.reduce_tile_for_backward(
        (x[:, :, :, :3], left_logits),
        left_valid,
        global_context,
    )
    replay_right = stream_reducer.reduce_tile_for_backward(
        (x[:, :, :, 1:], right_logits),
        right_valid,
        global_context,
    )
    torch.autograd.backward((replay_left, replay_right), (upstream, upstream))

    assert ref_bias.grad is not None
    assert stream_bias.grad is not None
    assert torch.allclose(stream_bias.grad, ref_bias.grad, atol=2e-6, rtol=2e-5)
    assert stream_bias.grad.abs().max() < 2e-6


def test_public_output_indices_skip_attention_gem_aux_payloads():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn._tile_output_shapes = [torch.Size((1, 1, 1, 1)) for _ in range(8)]
    scnn._reducer_head_map = {0: object(), 2: object(), 4: object(), 6: object()}
    scnn._reducer_input_indices = {0: (0, 1), 2: (2, 3), 4: (4, 5), 6: (6, 7)}
    scnn._output_spec = ("tuple", [("tensor", None) for _ in range(4)])

    public_indices = scnn._public_output_indices()
    expected_flat_outputs = scnn._count_tensors_in_spec(scnn._output_spec)
    outputs = [f"output_{idx}" for idx in range(8)]

    assert public_indices == [0, 2, 4, 6]
    assert len(public_indices) == expected_flat_outputs
    assert [outputs[idx] for idx in public_indices] == [
        "output_0",
        "output_2",
        "output_4",
        "output_6",
    ]


def test_prepare_forward_outputs_skips_reducer_aux_payloads():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn._tile_output_shapes = [
        torch.Size((1, 3, 2, 2)),
        torch.Size((1, 1, 2, 2)),
        torch.Size((1, 2, 2, 2)),
    ]
    scnn._reducer_head_map = {0: object()}
    scnn._reducer_input_indices = {0: (0, 1)}
    scnn.dtype = torch.float32

    image = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    outputs, allocate_non_reducer_outputs = StreamingCNN._prepare_forward_outputs(
        scnn,
        image=image,
        output_heights=[1, 1, 1],
        output_widths=[1, 1, 1],
        result_device=torch.device("cpu"),
    )

    allocate_non_reducer_outputs()

    assert outputs[0] is None
    assert outputs[1] is None
    assert outputs[2] is not None
    assert outputs[2].shape == (1, 2, 1, 1)
    assert torch.equal(outputs[2], torch.full((1, 2, 1, 1), 999.0))

    existing_output = outputs[2]
    allocate_non_reducer_outputs()
    assert outputs[2] is existing_output


def test_public_output_index_validation_rejects_reducer_aux_leak():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn._reducer_head_map = {0: object()}
    scnn._reducer_input_indices = {0: (0, 1)}

    with pytest.raises(RuntimeError) as exc_info:
        scnn._validate_public_output_indices([0, 1])

    message = str(exc_info.value)
    assert "public_indices=[0, 1]" in message
    assert "reducer_auxiliary_indices=[1]" in message
    assert "self._reducer_input_indices={0: (0, 1)}" in message


def test_public_forward_output_validation_reports_none_with_reducer_context():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn._reducer_head_map = {0: object()}
    scnn._reducer_input_indices = {0: (0, 1)}
    outputs = [torch.ones((1, 1, 1, 1)), None]

    with pytest.raises(RuntimeError) as exc_info:
        scnn._validate_public_forward_outputs(outputs, [0, 1])

    message = str(exc_info.value)
    assert "Public output head 1 was not populated" in message
    assert "public_indices=[0, 1]" in message
    assert "reducer_auxiliary_indices=[1]" in message
    assert "self._reducer_input_indices={0: (0, 1)}" in message


def test_public_forward_output_sentinel_check_is_debug_only():
    scnn = StreamingCNN.__new__(StreamingCNN)
    scnn._reducer_head_map = {}
    scnn._reducer_input_indices = {}
    outputs = [torch.full((1, 1, 1, 1), 999.0)]

    scnn._validate_public_forward_outputs(outputs, [0])

    scnn.debug_forward_sentinel_check = True
    with pytest.raises(RuntimeError) as exc_info:
        scnn._validate_public_forward_outputs(outputs, [0])

    message = str(exc_info.value)
    assert "unstitched sentinel value 999" in message
    assert "public_indices=[0]" in message
    assert "reducer_auxiliary_indices=[]" in message
    assert "self._reducer_input_indices={}" in message


def test_streaming_wss_attention_gem_public_outputs_skip_aux_maps(tmp_path):
    torch.manual_seed(17)
    image = torch.rand(1, 3, 80, 80)
    network = StreamingWSS(
        "resnet18",
        tile_size=64,
        weights=None,
        verbose=False,
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        tile_cache_path=tmp_path / "resnet18_wss_attention_gem_tile_cache",
    ).eval()
    scnn = network.stream_network
    scnn.debug_forward_sentinel_check = True

    with torch.no_grad():
        streamed = network(image)
        public_indices = scnn._public_output_indices()
        auxiliary_indices = scnn._reducer_aux_indices()

        scnn.disable()
        expected = scnn.stream_module(image)

    assert isinstance(streamed, tuple)
    assert len(streamed) == 4
    assert len(public_indices) == 4
    assert set(public_indices).isdisjoint(auxiliary_indices)
    assert sorted(auxiliary_indices) == [1, 3, 5]

    for streamed_reduced, expected_reduced in zip(streamed[:3], expected[:3]):
        assert streamed_reduced.shape == expected_reduced.shape
        assert list(streamed_reduced.shape) == [image.shape[0], expected_reduced.shape[1], 1, 1]

    assert streamed[3].shape == expected[3].shape

    for output in streamed:
        assert not torch.all(output == 999)

    for streamed_output, expected_output in zip(streamed, expected):
        assert torch.allclose(streamed_output, expected_output, atol=2e-4, rtol=2e-3)


def test_streaming_wss_attention_gem_backward_public_outputs_drive_aux_attention_grads(tmp_path):
    torch.manual_seed(19)
    image = torch.rand(1, 3, 80, 80)
    network = StreamingWSS(
        "resnet18",
        tile_size=64,
        weights=None,
        verbose=False,
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=True,
        normalize_on_gpu=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        tile_cache_path=tmp_path / "resnet18_wss_attention_gem_backward_tile_cache",
    ).eval()
    scnn = network.stream_network
    reference = copy.deepcopy(scnn.stream_module).eval()

    ref_image = image.detach().clone().requires_grad_(True)
    ref_outputs = reference(ref_image)
    ref_loss = (
        0.17 * ref_outputs[0].sum()
        - 0.23 * ref_outputs[1].mean()
        + 0.31 * ref_outputs[2].sum()
        + 0.07 * ref_outputs[3].mean()
    )
    ref_loss.backward()

    streamed = _require_and_retain_output_grads(network(image.detach().clone()))
    assert isinstance(streamed, tuple)
    assert len(streamed) == 4

    stream_loss = (
        0.17 * streamed[0].sum()
        - 0.23 * streamed[1].mean()
        + 0.31 * streamed[2].sum()
        + 0.07 * streamed[3].mean()
    )
    stream_loss.backward()
    assert all(output.grad is not None for output in streamed)
    public_grads = tuple(output.grad.detach().clone() for output in streamed)

    scnn.backward(image.detach().clone(), public_grads)

    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}
    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}

    selected_names = (
        "decoder1.0.weight",
        "decoder2.0.weight",
        "decoder3.0.weight",
        "att1.0.weight",
        "att2.0.weight",
        "att3.0.weight",
        "backbone.m.conv1.weight",
    )
    for name in selected_names:
        _assert_grad_matches(stream_grads, ref_grads, name)

    for name in ("att1.0.weight", "att2.0.weight", "att3.0.weight"):
        assert stream_grads[name].abs().sum() > 0, name


def test_streaming_attention_gem_backward_parity_x_and_logits():
    torch.manual_seed(13)
    model = AttentionGeMHeadNet().eval()
    reference = AttentionGeMHeadNet().eval()
    reference.load_state_dict(model.state_dict())

    image = (torch.rand(1, 3, 9, 11) + 0.05).requires_grad_(True)
    ref_image = image.detach().clone().requires_grad_(True)
    grad_out = torch.full((1, 3, 1, 1), 0.41, dtype=image.dtype)

    ref_out = reference(ref_image)
    torch.autograd.backward(ref_out, grad_out)

    scnn = _make_streaming(model, tile_size=4)
    scnn.debug_reducer_replay = True
    assert isinstance(scnn.stream_module.reducer, StreamingAttentionGeMReducer)
    stream_out = scnn.forward(image.detach().clone())
    assert torch.allclose(stream_out, ref_out.detach(), atol=1e-5, rtol=1e-4)
    scnn.backward(image.detach().clone(), grad_out.detach().clone())

    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}
    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}

    for name in ("feat_head.weight", "logit_head.weight"):
        assert name in stream_grads
        assert name in ref_grads
        assert torch.allclose(stream_grads[name], ref_grads[name], atol=2e-5, rtol=2e-4), name


def test_streaming_attention_gem_logit_bias_gradient_matches_reference():
    torch.manual_seed(23)
    model = AttentionGeMBiasHeadNet().eval()
    reference = AttentionGeMBiasHeadNet().eval()
    reference.load_state_dict(model.state_dict())

    image = (torch.rand(1, 3, 9, 11) + 0.05).requires_grad_(True)
    ref_image = image.detach().clone().requires_grad_(True)
    grad_out = torch.tensor([[[[0.19]], [[-0.31]], [[0.43]]]], dtype=image.dtype)

    ref_out = reference(ref_image)
    torch.autograd.backward(ref_out, grad_out)

    scnn = _make_streaming(model, tile_size=4)
    scnn.debug_reducer_replay = True
    stream_out = scnn.forward(image.detach().clone())
    assert torch.allclose(stream_out, ref_out.detach(), atol=1e-5, rtol=1e-4)
    scnn.backward(image.detach().clone(), grad_out.detach().clone())

    stream_bias_grad = scnn.stream_module.logit_head.bias.grad
    ref_bias_grad = reference.logit_head.bias.grad

    assert stream_bias_grad is not None
    assert ref_bias_grad is not None
    assert torch.allclose(stream_bias_grad, ref_bias_grad, atol=2e-6, rtol=2e-5)
    assert stream_bias_grad.abs().max() < 2e-6
