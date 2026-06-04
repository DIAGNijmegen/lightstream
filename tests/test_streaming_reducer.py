import copy

import torch
import torch.nn as nn
import pytest

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.models.segment.streamingwss import StreamingWSS

from lightstream.core.reducer import (
    AttentionGeMReducer,
    FusedAttentionGeMReducer,
    GeMReducer,
    MeanReducer,
    StreamingAttentionGeMReducer,
    StreamingFusedAttentionGeMReducer,
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


class SmallGatedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid_branch = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1), nn.Sigmoid())
        self.tanh_branch = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1), nn.Tanh())
        self.att_logits = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.att_logits(self.sigmoid_branch(x) * self.tanh_branch(x))


class SmallWSSLikeReducerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(4, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(4, 3, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )
        self.att1 = SmallGatedAttention()
        self.att2 = SmallGatedAttention()
        self.att3 = SmallGatedAttention()
        self.red1 = AttentionGeMReducer(r_init=2.4, eps=1e-6)
        self.red2 = AttentionGeMReducer(r_init=2.4, eps=1e-6)
        self.red3 = AttentionGeMReducer(r_init=2.4, eps=1e-6)
        self.red4 = FusedAttentionGeMReducer(
            r_init=2.6,
            eps=1e-6,
            value_weights=(0.3, 0.4, 0.3),
            attention_weights=(0.3, 0.4, 0.3),
        )

    def forward(self, x, mask: torch.Tensor | None = None):
        feat = self.stem(x)
        y1 = self.decoder1(feat)
        y2 = self.decoder2(feat)
        y3 = self.decoder3(feat)
        att1 = self.att1(y1)
        att2 = self.att2(y2)
        att3 = self.att3(y3)
        return (
            self.red1(y1, att1, mask=mask),
            self.red2(y2, att2, mask=mask),
            self.red3(y3, att3, mask=mask),
            self.red4(y1, y2, y3, att1, att2, att3, mask=mask),
        )


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


def test_streaming_attention_gem_masked_forward_and_backward_replay_match_non_streaming():
    torch.manual_seed(43)
    image = torch.randn(1, 3, 4, 6, dtype=torch.float32)
    x_weight = torch.randn(2, 3, 1, 1, dtype=torch.float32) * 0.2
    logit_weight = torch.randn(1, 3, 1, 1, dtype=torch.float32) * 0.3
    logit_bias = torch.tensor([0.17], dtype=torch.float32)
    mask = torch.tensor(
        [
            [1, 0, 1, 1, 0, 1],
            [0, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    invalid_logit_boost = (~mask).to(dtype=image.dtype)[None, None] * 8.0
    upstream = torch.tensor([[[[0.23]], [[-0.37]]]], dtype=image.dtype)
    sides = type("S", (), dict(top=False, left=False, right=False, bottom=False))()
    tiles = (
        (slice(0, 2), slice(0, 3), (0, 2, 0, 3)),
        (slice(0, 2), slice(3, 6), (0, 2, 3, 6)),
        (slice(2, 4), slice(0, 3), (2, 4, 0, 3)),
        (slice(2, 4), slice(3, 6), (2, 4, 3, 6)),
    )

    ref_image = image.detach().clone().requires_grad_(True)
    ref_x_weight = x_weight.detach().clone().requires_grad_(True)
    ref_logit_weight = logit_weight.detach().clone().requires_grad_(True)
    ref_logit_bias = logit_bias.detach().clone().requires_grad_(True)
    ref_x = torch.nn.functional.conv2d(ref_image, ref_x_weight).sigmoid() + 0.05
    ref_x.retain_grad()
    ref_logits = torch.nn.functional.conv2d(ref_image, ref_logit_weight, ref_logit_bias) + invalid_logit_boost
    ref_logits.retain_grad()
    ref_reducer = AttentionGeMReducer(r_init=2.4, eps=1e-6)
    ref_out = ref_reducer(ref_x, ref_logits, mask=mask)
    torch.autograd.backward(ref_out, upstream)

    stream_image = image.detach().clone().requires_grad_(True)
    stream_x_weight = x_weight.detach().clone().requires_grad_(True)
    stream_logit_weight = logit_weight.detach().clone().requires_grad_(True)
    stream_logit_bias = logit_bias.detach().clone().requires_grad_(True)
    stream_x = torch.nn.functional.conv2d(stream_image, stream_x_weight).sigmoid() + 0.05
    stream_x.retain_grad()
    stream_logits = torch.nn.functional.conv2d(stream_image, stream_logit_weight, stream_logit_bias) + invalid_logit_boost
    stream_logits.retain_grad()
    stream_reducer = StreamingAttentionGeMReducer(r_init=2.4, eps=1e-6)
    stream_reducer.start_stream(
        output_height=image.shape[-2],
        output_width=image.shape[-1],
        batch_size=image.shape[0],
        channels=stream_x.shape[1],
        device=image.device,
        dtype=image.dtype,
    )

    with torch.no_grad():
        for tile_idx, (ys, xs, dst_box) in enumerate(tiles):
            stream_reducer.accumulate_stream_tile(
                (stream_x.detach()[:, :, ys, xs], stream_logits.detach()[:, :, ys, xs]),
                tile_idx // 2,
                tile_idx % 2,
                sides,
                dst_box,
                user_mask=mask[ys, xs],
            )
        stream_out = stream_reducer.finish_stream()

    assert torch.allclose(stream_out, ref_out.detach(), atol=1e-6, rtol=1e-5)

    global_context = stream_reducer.extra_state_for_backward()
    replay_outputs = []
    replay_grads = []
    for ys, xs, _dst_box in tiles:
        replay_outputs.append(
            stream_reducer.reduce_tile_for_backward(
                (stream_x[:, :, ys, xs], stream_logits[:, :, ys, xs]),
                mask[ys, xs],
                global_context,
            )
        )
        replay_grads.append(upstream)
    torch.autograd.backward(tuple(replay_outputs), tuple(replay_grads))

    assert torch.allclose(stream_image.grad, ref_image.grad, atol=2e-5, rtol=2e-4)
    assert torch.allclose(stream_logits.grad, ref_logits.grad, atol=2e-5, rtol=2e-4)
    assert torch.allclose(stream_x.grad, ref_x.grad, atol=2e-5, rtol=2e-4)
    assert torch.allclose(stream_x_weight.grad, ref_x_weight.grad, atol=2e-5, rtol=2e-4)
    assert torch.allclose(stream_logit_weight.grad, ref_logit_weight.grad, atol=2e-5, rtol=2e-4)
    assert torch.allclose(stream_logit_bias.grad, ref_logit_bias.grad, atol=2e-5, rtol=2e-4)


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


def _install_common_crop_recorder(scnn: StreamingCNN):
    records = []
    original = scnn._build_common_aligned_reducer_payload

    def recording_build_common_aligned_reducer_payload(
        *, head_idx, tile_outputs, ordered_indices, tile_input_y, tile_input_x, sides
    ):
        result = original(
            head_idx=head_idx,
            tile_outputs=tile_outputs,
            ordered_indices=ordered_indices,
            tile_input_y=tile_input_y,
            tile_input_x=tile_input_x,
            sides=sides,
        )
        _trimmed_payload, _common_loc, common_dst_box = result
        records.append(
            {
                "head_idx": int(head_idx),
                "indices": tuple(int(idx) for idx in ordered_indices),
                "tile": (int(tile_input_y), int(tile_input_x)),
                "common_dst_box": tuple(int(v) for v in common_dst_box),
            }
        )
        return result

    scnn._build_common_aligned_reducer_payload = recording_build_common_aligned_reducer_payload
    return records


def test_streaming_wss_like_branch_gradients_match_non_streaming_with_masked_lost_borders():
    torch.manual_seed(211)
    model = SmallWSSLikeReducerNet().eval()
    reference = SmallWSSLikeReducerNet().eval()
    reference.load_state_dict(model.state_dict())

    image = torch.randn(1, 3, 32, 36)
    mask = (torch.arange(32)[:, None] * 3 + torch.arange(36)[None, :] * 5) % 7 != 0

    ref_image = image.detach().clone().requires_grad_(True)
    ref_outputs = reference(ref_image, mask=mask)
    grad_outputs = (
        torch.full_like(ref_outputs[0], 0.17),
        torch.full_like(ref_outputs[1], -0.23),
        torch.full_like(ref_outputs[2], 0.31),
        torch.full_like(ref_outputs[3], 0.07),
    )
    torch.autograd.backward(ref_outputs, grad_outputs)
    ref_grads = {
        name: p.grad.detach().clone()
        for name, p in reference.named_parameters()
        if p.grad is not None
    }

    scnn = _make_streaming(model, tile_size=20)
    scnn.debug_reducer_replay = True
    common_crop_records = _install_common_crop_recorder(scnn)
    assert scnn.tile_gradient_lost.top > 0
    assert scnn.tile_gradient_lost.left > 0
    assert scnn.tile_gradient_lost.bottom > 0
    assert scnn.tile_gradient_lost.right > 0

    stream_outputs = scnn.forward(image.detach().clone(), mask=mask)
    assert isinstance(stream_outputs, tuple)
    assert len(stream_outputs) == 4
    for stream_output, ref_output in zip(stream_outputs, ref_outputs):
        assert torch.allclose(stream_output, ref_output.detach(), atol=2e-5, rtol=2e-4)

    scnn.backward(image.detach().clone(), tuple(grad.detach().clone() for grad in grad_outputs), mask=mask)
    stream_grads = {
        name: p.grad.detach().clone()
        for name, p in scnn.stream_module.named_parameters()
        if p.grad is not None
    }

    compared_names = [
        name
        for name, _param in reference.named_parameters()
        if name.startswith(("decoder1.", "decoder2.", "decoder3.", "att1.", "att2.", "att3."))
    ]
    assert compared_names
    assert common_crop_records

    crop_debug = (
        f"reducer_input_indices={scnn._reducer_input_indices}; "
        f"common_crop_boxes={common_crop_records}"
    )
    for name in compared_names:
        assert name in ref_grads, f"missing reference gradient for {name}; {crop_debug}"
        assert name in stream_grads, f"missing streaming gradient for {name}; {crop_debug}"
        assert torch.isfinite(ref_grads[name]).all(), f"non-finite reference gradient for {name}; {crop_debug}"
        assert torch.isfinite(stream_grads[name]).all(), f"non-finite streaming gradient for {name}; {crop_debug}"
        if not torch.allclose(stream_grads[name], ref_grads[name], atol=2e-5, rtol=2e-4):
            max_abs = (stream_grads[name] - ref_grads[name]).abs().max().item()
            pytest.fail(f"gradient mismatch for {name}: max_abs={max_abs}; {crop_debug}")


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


class FusedAttentionGeMHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False), nn.ReLU())
        self.value1 = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.value2 = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.value3 = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.logits1 = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.logits2 = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.logits3 = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.reducer = FusedAttentionGeMReducer(
            r_init=2.6,
            eps=1e-6,
            value_weights=(0.2, 0.5, 0.3),
            attention_weights=(0.6, 0.1, 0.3),
        )

    def forward(self, x, mask: torch.Tensor | None = None):
        feat = self.backbone(x)
        y1 = self.value1(feat).sigmoid() + 0.05
        y2 = self.value2(feat).sigmoid() + 0.05
        y3 = self.value3(feat).sigmoid() + 0.05
        l1 = self.logits1(feat)
        l2 = self.logits2(feat)
        l3 = self.logits3(feat)
        return self.reducer(y1, y2, y3, l1, l2, l3, mask=mask), y1, y2, y3, l1, l2, l3


def _fused_attention_gem_reference(
    y1,
    y2,
    y3,
    logits1,
    logits2,
    logits3,
    *,
    r=4.0,
    eps=1e-6,
    value_weights=(0.3, 0.4, 0.3),
    attention_weights=(0.3, 0.4, 0.3),
    mask=None,
):
    acc_dtype = torch.float64
    values = [y.to(acc_dtype) for y in (y1, y2, y3)]
    fused_y = sum(float(w) * y for w, y in zip(value_weights, values))
    x_pow = fused_y.clamp_min(eps).pow(torch.tensor(float(r), dtype=acc_dtype, device=y1.device))
    branch_means = []
    for logits in (logits1, logits2, logits3):
        logits = logits.to(acc_dtype)
        if logits.ndim == 3:
            logits = logits[:, None]
        elif logits.shape[1] == y1.shape[1]:
            logits = logits.mean(dim=1, keepdim=True)
        if mask is not None:
            mask4 = mask.to(device=y1.device, dtype=torch.bool)
            if mask4.ndim == 2:
                mask4 = mask4[None, None]
            elif mask4.ndim == 3:
                mask4 = mask4[:, None]
            logits = torch.where(mask4, logits, torch.full_like(logits, torch.finfo(acc_dtype).min))
        weights = torch.softmax(logits.flatten(-2), dim=-1).view_as(logits)
        if mask is not None:
            weights = torch.where(mask4, weights, torch.zeros_like(weights))
        branch_means.append((weights * x_pow).sum(dim=(-2, -1), keepdim=True))
    weighted_mean = sum(float(w) * mean for w, mean in zip(attention_weights, branch_means))
    return weighted_mean.clamp_min(eps).pow(1.0 / float(r)).to(y1.dtype)


def test_fused_attention_gem_forward_reference_unmasked_and_masked_custom_weights():
    torch.manual_seed(201)
    y1 = torch.rand(2, 3, 5, 7) + 0.05
    y2 = torch.rand(2, 3, 5, 7) + 0.05
    y3 = torch.rand(2, 3, 5, 7) + 0.05
    logits = [torch.randn(2, 1, 5, 7), torch.randn(2, 3, 5, 7), torch.randn(2, 5, 7)]
    reducer = FusedAttentionGeMReducer(
        r_init=2.25,
        eps=1e-6,
        value_weights=(0.15, 0.55, 0.30),
        attention_weights=(0.50, 0.20, 0.30),
        accumulator_dtype=torch.float64,
    )

    actual = reducer(y1, y2, y3, *logits)
    expected = _fused_attention_gem_reference(
        y1,
        y2,
        y3,
        *logits,
        r=2.25,
        value_weights=(0.15, 0.55, 0.30),
        attention_weights=(0.50, 0.20, 0.30),
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    mask = (torch.arange(5)[:, None] + torch.arange(7)[None, :]) % 4 != 0
    actual_masked = reducer(y1, y2, y3, *logits, mask=mask)
    expected_masked = _fused_attention_gem_reference(
        y1,
        y2,
        y3,
        *logits,
        r=2.25,
        value_weights=(0.15, 0.55, 0.30),
        attention_weights=(0.50, 0.20, 0.30),
        mask=mask,
    )
    assert torch.allclose(actual_masked, expected_masked, atol=1e-6, rtol=1e-5)


def test_fused_attention_gem_to_streaming_copies_state():
    reducer = FusedAttentionGeMReducer(
        r_init=3.7,
        eps=1e-5,
        value_weights=(0.1, 0.2, 0.7),
        attention_weights=(0.7, 0.2, 0.1),
        accumulator_dtype=torch.float64,
    )
    streaming = reducer.to_streaming()
    assert isinstance(streaming, StreamingFusedAttentionGeMReducer)
    assert torch.equal(streaming.r, reducer.r)
    assert streaming.eps == reducer.eps
    assert torch.equal(streaming.value_weights, reducer.value_weights)
    assert torch.equal(streaming.attention_weights, reducer.attention_weights)
    assert streaming.accumulator_dtype == torch.float64


def test_streaming_fused_attention_gem_forward_parity_odd_masked_tiles():
    torch.manual_seed(203)
    y1 = torch.rand(1, 2, 5, 7) + 0.05
    y2 = torch.rand(1, 2, 5, 7) + 0.05
    y3 = torch.rand(1, 2, 5, 7) + 0.05
    logits = [torch.randn(1, 1, 5, 7), torch.randn(1, 1, 5, 7), torch.randn(1, 1, 5, 7)]
    mask = (torch.arange(5)[:, None] * 2 + torch.arange(7)[None, :]) % 5 != 0
    reducer = FusedAttentionGeMReducer(r_init=2.8, value_weights=(0.2, 0.3, 0.5), attention_weights=(0.5, 0.25, 0.25))
    expected = reducer(y1, y2, y3, *logits, mask=mask)

    stream = reducer.to_streaming()
    stream.start_stream(output_height=5, output_width=7, batch_size=1, channels=2, device=y1.device, dtype=y1.dtype)
    sides = type("S", (), dict(top=False, left=False, right=False, bottom=False))()
    for y0, y1s in ((0, 3), (3, 5)):
        for x0, x1s in ((0, 4), (4, 7)):
            tile = tuple(t[:, :, y0:y1s, x0:x1s] for t in (y1, y2, y3, *logits))
            stream.accumulate_stream_tile(tile, y0, x0, sides, (y0, y1s, x0, x1s), user_mask=mask[y0:y1s, x0:x1s])
    actual = stream.finish_stream()
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)




def _scnn_like_reducer_tiles(height: int, width: int, tile_height: int, tile_width: int, step_y: int, step_x: int):
    """Yield reducer-domain tile windows with SCNN bottom/right repositioning."""
    import math

    n_rows = math.ceil(float(max(1, height - tile_height)) / float(step_y)) + 1
    n_cols = math.ceil(float(max(1, width - tile_width)) / float(step_x)) + 1
    if height <= tile_height:
        n_rows = 1
    if width <= tile_width:
        n_cols = 1

    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * step_y
            x0 = col * step_x
            bottom = y0 + tile_height >= height
            right = x0 + tile_width >= width
            if bottom:
                y0 = max(height - tile_height, 0)
            if right:
                x0 = max(width - tile_width, 0)
            if row == 0:
                y0 = 0
            if col == 0:
                x0 = 0
            y1 = min(y0 + tile_height, height)
            x1 = min(x0 + tile_width, width)
            sides = type("S", (), dict(top=row == 0, left=col == 0, right=right, bottom=bottom))()
            yield y0, y1, x0, x1, sides


def test_streaming_fused_attention_gem_reduce_tile_for_backward_scnn_like_tiles_all_inputs():
    torch.manual_seed(209)
    dtype = torch.float64
    batch, channels, height, width = 1, 2, 48, 50
    tile_height, tile_width = 25, 25
    step_y, step_x = 16, 16
    value_weights = (0.2, 0.5, 0.3)
    attention_weights = (0.6, 0.1, 0.3)
    r_init = 2.6
    eps = 1e-6

    base_inputs = {
        "y1": torch.rand(batch, channels, height, width, dtype=dtype) + 0.05,
        "y2": torch.rand(batch, channels, height, width, dtype=dtype) + 0.05,
        "y3": torch.rand(batch, channels, height, width, dtype=dtype) + 0.05,
        "logits1": torch.randn(batch, 1, height, width, dtype=dtype),
        "logits2": torch.randn(batch, 1, height, width, dtype=dtype),
        "logits3": torch.randn(batch, 1, height, width, dtype=dtype),
    }
    mask = (torch.rand(height, width) > 0.17)
    upstream = torch.tensor([[[[0.37]], [[-0.19]]]], dtype=dtype)

    ref_inputs = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in base_inputs.items()}
    ref_reducer = FusedAttentionGeMReducer(
        r_init=r_init,
        eps=eps,
        value_weights=value_weights,
        attention_weights=attention_weights,
        accumulator_dtype=torch.float64,
    )
    ref_out = ref_reducer(
        ref_inputs["y1"],
        ref_inputs["y2"],
        ref_inputs["y3"],
        ref_inputs["logits1"],
        ref_inputs["logits2"],
        ref_inputs["logits3"],
        mask=mask,
    )
    torch.autograd.backward(ref_out, upstream)
    ref_grads = {name: tensor.grad.detach().clone() for name, tensor in ref_inputs.items()}

    stream_inputs = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in base_inputs.items()}
    stream_reducer = StreamingFusedAttentionGeMReducer(
        r_init=r_init,
        eps=eps,
        value_weights=value_weights,
        attention_weights=attention_weights,
        accumulator_dtype=torch.float64,
    )
    stream_reducer.start_stream(
        output_height=height,
        output_width=width,
        batch_size=batch,
        channels=channels,
        device=base_inputs["y1"].device,
        dtype=dtype,
    )

    tiles = list(_scnn_like_reducer_tiles(height, width, tile_height, tile_width, step_y, step_x))
    assert len(tiles) == 9
    assert tiles[-1][:4] == (height - tile_height, height, width - tile_width, width)

    with torch.no_grad():
        for tile_idx, (y0, y1, x0, x1, sides) in enumerate(tiles):
            forward_tile = tuple(
                base_inputs[name][:, :, y0:y1, x0:x1]
                for name in ("y1", "y2", "y3", "logits1", "logits2", "logits3")
            )
            stream_reducer.accumulate_stream_tile(
                forward_tile,
                tile_idx // 3,
                tile_idx % 3,
                sides,
                (y0, y1, x0, x1),
                user_mask=mask[y0:y1, x0:x1],
            )
        stream_out = stream_reducer.finish_stream()

    assert torch.allclose(stream_out, ref_out.detach(), atol=1e-12, rtol=1e-12)

    stream_reducer.start_backward_replay()
    seen = torch.zeros((height, width), dtype=torch.bool)
    replay_outputs = []
    replay_grads = []
    for tile_idx, (y0, y1, x0, x1, sides) in enumerate(tiles):
        dst_box = (y0, y1, x0, x1)
        dst_y0, dst_y1, dst_x0, dst_x1 = dst_box
        new_mask = ~seen[dst_y0:dst_y1, dst_x0:dst_x1]
        tile_user_mask = mask[dst_y0:dst_y1, dst_x0:dst_x1]
        effective_mask = new_mask & tile_user_mask
        backward_tile = tuple(
            stream_inputs[name][:, :, y0:y1, x0:x1]
            for name in ("y1", "y2", "y3", "logits1", "logits2", "logits3")
        )
        replay_output, replay_grad = stream_reducer.build_backward_pair(
            backward_tile,
            upstream,
            input_y=tile_idx // 3,
            input_x=tile_idx % 3,
            sides=sides,
            valid_mask=effective_mask,
        )
        replay_outputs.append(replay_output)
        replay_grads.append(replay_grad)
        seen[dst_y0:dst_y1, dst_x0:dst_x1] |= new_mask

    torch.autograd.backward(tuple(replay_outputs), tuple(replay_grads))
    stream_reducer.validate_backward_replay_consumed(head_idx=0)

    diagnostic_atol = 1e-8
    diagnostic_rtol = 1e-7
    strict_atol = 5e-10
    strict_rtol = 5e-9
    input_names = ("y1", "y2", "y3", "logits1", "logits2", "logits3")
    diagnostics = [
        "Gradient diagnostics for streaming fused attention GeM backward replay",
        f"diagnostic threshold: atol={diagnostic_atol:g}, rtol={diagnostic_rtol:g}",
        f"strict threshold retained for follow-up: atol={strict_atol:g}, rtol={strict_rtol:g}",
    ]
    failures = []

    for name in input_names:
        stream_grad = stream_inputs[name].grad
        ref_grad = ref_grads[name]
        if stream_grad is None:
            diagnostics.append(f"{name}: missing streaming gradient")
            failures.append(name)
            continue

        diff = (stream_grad - ref_grad).abs()
        finite = bool(torch.isfinite(stream_grad).all().item())
        mean_abs = diff.mean().item()
        max_abs = diff.max().item()
        max_flat_idx = torch.argmax(diff).item()
        max_idx = tuple(
            int(i)
            for i in torch.unravel_index(torch.tensor(max_flat_idx, device=diff.device), diff.shape)
        )
        stream_value = stream_grad[max_idx].item()
        ref_value = ref_grad[max_idx].item()
        rtol_denominator = diagnostic_rtol * ref_grad.abs().clamp_min(torch.finfo(ref_grad.dtype).tiny)
        rtol_scaled_max = (diff / rtol_denominator).max().item()
        close = torch.allclose(stream_grad, ref_grad, atol=diagnostic_atol, rtol=diagnostic_rtol)

        diagnostics.append(
            f"{name}: close={close}, finite={finite}, "
            f"mean_abs={mean_abs:.17g}, max_abs={max_abs:.17g}, "
            f"rtol-scaled max={rtol_scaled_max:.17g}, max_idx={max_idx}, "
            f"stream={stream_value:.17g}, ref={ref_value:.17g}"
        )
        if not finite or not close:
            failures.append(name)

    assert not failures, "\n".join(diagnostics + [f"failed inputs: {', '.join(failures)}"])


def test_scnn_fused_attention_gem_conversion_and_public_outputs_skip_aux_payloads():
    torch.manual_seed(205)
    model = FusedAttentionGeMHeadNet().eval()
    image = torch.randn(1, 3, 9, 11)
    scnn = _make_streaming(model, tile_size=4)
    assert isinstance(scnn.stream_module.reducer, StreamingFusedAttentionGeMReducer)

    with torch.no_grad():
        streamed = scnn.forward(image)
        expected = model(image)

    assert isinstance(streamed, tuple)
    assert len(streamed) == 7
    assert sorted(scnn._reducer_aux_indices()) == [1, 2, 3, 4, 5]
    assert torch.allclose(streamed[0], expected[0], atol=1e-5, rtol=1e-4)
    for streamed_aux, expected_aux in zip(streamed[1:], expected[1:]):
        assert torch.allclose(streamed_aux, expected_aux, atol=1e-5, rtol=1e-4)


def test_streaming_fused_attention_gem_backward_parity_all_inputs():
    torch.manual_seed(207)
    model = FusedAttentionGeMHeadNet().eval()
    reference = FusedAttentionGeMHeadNet().eval()
    reference.load_state_dict(model.state_dict())
    image = torch.randn(1, 3, 7, 9)
    mask = (torch.rand(7, 9) > 0.2)

    ref_image = image.detach().clone().requires_grad_(True)
    ref_outputs = reference(ref_image, mask=mask)
    grad_reduced = torch.tensor([[[[0.37]], [[-0.19]]]], dtype=image.dtype)
    aux_grads = tuple(torch.zeros_like(output) for output in ref_outputs[1:])
    torch.autograd.backward(ref_outputs, (grad_reduced, *aux_grads))
    ref_grads = {name: p.grad.detach().clone() for name, p in reference.named_parameters() if p.grad is not None}

    scnn = _make_streaming(model, tile_size=4)
    scnn.debug_reducer_replay = True
    stream_outputs = scnn.forward(image.detach().clone(), mask=mask)
    assert torch.allclose(stream_outputs[0], ref_outputs[0].detach(), atol=1e-5, rtol=1e-4)
    scnn.backward(image.detach().clone(), (grad_reduced, *(torch.zeros_like(output) for output in stream_outputs[1:])), mask=mask)
    stream_grads = {name: p.grad for name, p in scnn.stream_module.named_parameters() if p.grad is not None}
    for name, ref_grad in ref_grads.items():
        assert name in stream_grads
        assert torch.allclose(stream_grads[name], ref_grad, atol=4e-5, rtol=4e-4), name


class RecordingBackwardReducer:
    def __init__(self):
        self.valid_masks = []

    def build_backward_pair(
        self,
        payload,
        gradient,
        *,
        input_y,
        input_x,
        sides,
        valid_mask,
    ):
        self.valid_masks.append(valid_mask.detach().clone())
        return payload, gradient


def _backward_mask_replay_attr_names():
    replay = "replay"
    effective = "effective"
    mask = "mask"
    masks = f"{mask}s"
    cursor = "cursor"
    record = "record"
    consume = "consume"
    backward = "backward"
    return (
        f"_{replay}_{effective}_{masks}",
        f"_{replay}_{effective}_{mask}_{cursor}",
        f"_{record}_{effective}_{mask}_for_{backward}",
        f"_{consume}_{effective}_{mask}_for_{backward}",
    )


def test_streaming_reducers_do_not_keep_backward_mask_replay_state():
    reducers = (
        StreamingMeanReducer(),
        StreamingGeMReducer(),
        StreamingAttentionGeMReducer(),
        StreamingFusedAttentionGeMReducer(),
    )
    attr_names = _backward_mask_replay_attr_names()

    for reducer in reducers:
        assert all(not hasattr(reducer, name) for name in attr_names)
        reducer.start_stream(
            output_height=3,
            output_width=4,
            batch_size=1,
            channels=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            debug_replay=True,
        )
        reducer.start_backward_replay()
        assert all(not hasattr(reducer, name) for name in attr_names)


def test_scnn_backward_reducer_effective_masks_are_per_head_and_common_dst_box():
    scnn = StreamingCNN.__new__(StreamingCNN)
    reducer0 = RecordingBackwardReducer()
    reducer1 = RecordingBackwardReducer()
    scnn._reducer_head_map = {0: reducer0, 1: reducer1}
    scnn._reducer_input_indices = {}
    scnn._active_reducer_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    scnn._backward_reducer_seen_masks = {
        0: torch.zeros((4, 4), dtype=torch.bool),
        1: torch.zeros((4, 4), dtype=torch.bool),
    }
    scnn.device = torch.device("cpu")
    sides = type("S", (), dict(top=False, left=False, right=False, bottom=False))()
    gradient = torch.ones((1, 1, 1, 1))

    first_tile = torch.ones((1, 1, 2, 2))
    second_tile = torch.ones((1, 1, 2, 2))
    scnn._build_reducer_backward_pair(
        0, first_tile, [first_tile], gradient, 0, 0, sides, 0, 0
    )
    scnn._build_reducer_backward_pair(
        0, second_tile, [second_tile], gradient, 0, 0, sides, 1, 1
    )
    scnn._build_reducer_backward_pair(
        1, second_tile, [second_tile], gradient, 0, 0, sides, 1, 1
    )

    assert torch.equal(
        reducer0.valid_masks[0],
        torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
    )
    assert torch.equal(
        reducer0.valid_masks[1],
        torch.tensor([[0, 1], [1, 1]], dtype=torch.bool),
    )
    assert torch.equal(
        reducer1.valid_masks[0],
        torch.tensor([[0, 1], [1, 1]], dtype=torch.bool),
    )
    assert scnn._backward_reducer_seen_masks[0].sum().item() == 7
    assert scnn._backward_reducer_seen_masks[1].sum().item() == 4
