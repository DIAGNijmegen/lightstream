import torch
import torch.nn.functional as F

from lightstream.modules.loss_reducer import GlobalWSLossReducer, StreamingGlobalWSLossReducer


def _tile_slices(height: int, width: int, tile_h: int, tile_w: int):
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            yield y, x, slice(y, min(y + tile_h, height)), slice(x, min(x + tile_w, width))


def _reference_wsl_loss(logits: torch.Tensor, slide_label: torch.Tensor, r: float, spatial_mask: torch.Tensor | None = None):
    probs_r = torch.sigmoid(logits).pow(r)
    if spatial_mask is None:
        mean_p_r = probs_r.mean(dim=(-2, -1))
    else:
        mask = spatial_mask.to(device=probs_r.device, dtype=probs_r.dtype)[None, None, :, :]
        valid = int(spatial_mask.sum().item())
        mean_p_r = (probs_r * mask).sum(dim=(-2, -1)) / valid
    pooled = mean_p_r.pow(1.0 / r)
    return F.binary_cross_entropy(pooled, slide_label)


def test_streaming_ws_reducer_matches_non_streaming_loss_and_gradients():
    torch.manual_seed(0)
    r = 4.0
    logits_full = torch.randn((1, 1, 31, 37), requires_grad=True)
    slide_label = torch.tensor([[1.0]])

    ref_loss = _reference_wsl_loss(logits_full, slide_label, r)
    ref_loss.backward()
    ref_grad = logits_full.grad.detach().clone()

    logits_stream = logits_full.detach().clone().requires_grad_(True)
    reducer = StreamingGlobalWSLossReducer(r=r)
    reducer.reset()

    for _, _, ys, xs in _tile_slices(31, 37, 9, 10):
        reducer.update(logits_stream[:, :, ys, xs])

    stream_loss = reducer.finalize(slide_label)
    stream_loss.backward()

    assert torch.allclose(stream_loss.detach(), ref_loss.detach(), atol=1e-7, rtol=1e-6)
    assert torch.allclose(logits_stream.grad, ref_grad, atol=1e-7, rtol=1e-6)


def test_streaming_ws_reducer_handles_overlap_with_seen_indices_and_lost_crop():
    torch.manual_seed(1)
    r = 3.0
    logits = torch.randn((1, 1, 18, 20))
    slide_label = torch.tensor([[0.0]])

    # Reference will be computed on the exact set of covered pixels.

    # Simulate overlapping tiles with invalid borders that need cropping.
    reducer = StreamingGlobalWSLossReducer(r=r)
    reducer.reset(spatial_shape=(18, 20))

    tile_h, tile_w = 12, 14
    stride_h, stride_w = 9, 10
    # Crop away invalid border values before deduplication.
    lost = (1, 2, 1, 2)

    for y, x, _, _ in _tile_slices(18, 20, stride_h, stride_w):
        y0 = min(y, 18 - tile_h)
        x0 = min(x, 20 - tile_w)
        tile = logits[:, :, y0 : y0 + tile_h, x0 : x0 + tile_w]

        # origin must point to the post-crop area in output space
        origin = (y0 + lost[0], x0 + lost[1])
        reducer.update(tile, tile_origin=origin, lost=lost)

    stream_loss = reducer.finalize(slide_label)
    covered_mask = reducer.state.seen_indices
    ref_loss = _reference_wsl_loss(logits, slide_label, r, spatial_mask=covered_mask)
    assert torch.allclose(stream_loss.detach(), ref_loss.detach(), atol=1e-7, rtol=1e-6)


def test_global_ws_reducer_matches_reference_formula():
    torch.manual_seed(3)
    logits = torch.randn((2, 1, 8, 9))
    slide_label = torch.tensor([[1.0], [0.0]])
    reducer = GlobalWSLossReducer(r=5.0)

    loss = reducer(logits, slide_label)
    ref = _reference_wsl_loss(logits, slide_label, r=5.0)

    assert torch.allclose(loss.detach(), ref.detach(), atol=1e-7, rtol=1e-6)


def test_streaming_ws_reducer_overlap_gradients_match_masked_reference():
    torch.manual_seed(7)
    r = 3.0
    slide_label = torch.tensor([[1.0]])

    logits_stream = torch.randn((1, 1, 18, 20), requires_grad=True)
    reducer = StreamingGlobalWSLossReducer(r=r)
    reducer.reset(spatial_shape=(18, 20))

    tile_h, tile_w = 12, 14
    stride_h, stride_w = 9, 10
    lost = (1, 2, 1, 2)

    for y, x, _, _ in _tile_slices(18, 20, stride_h, stride_w):
        y0 = min(y, 18 - tile_h)
        x0 = min(x, 20 - tile_w)
        tile = logits_stream[:, :, y0 : y0 + tile_h, x0 : x0 + tile_w]
        origin = (y0 + lost[0], x0 + lost[1])
        reducer.update(tile, tile_origin=origin, lost=lost)

    loss_stream = reducer.finalize(slide_label)
    loss_stream.backward()
    grad_stream = logits_stream.grad.detach().clone()

    # Reference gradient on the exact covered set of pixels.
    logits_ref = logits_stream.detach().clone().requires_grad_(True)
    covered_mask = reducer.state.seen_indices
    ref_loss = _reference_wsl_loss(logits_ref, slide_label, r, spatial_mask=covered_mask)
    ref_loss.backward()
    grad_ref = logits_ref.grad.detach().clone()

    assert torch.allclose(loss_stream.detach(), ref_loss.detach(), atol=1e-7, rtol=1e-6)
    assert torch.allclose(grad_stream, grad_ref, atol=1e-7, rtol=1e-6)
