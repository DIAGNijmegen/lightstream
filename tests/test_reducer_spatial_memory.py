"""Regression checks for tile ownership without full-slide coverage storage."""

import torch
import pytest
from torch import nn

from lightstream.core.reducer.base import streaming_reduce_tile
from lightstream.core.reducer.mean import StreamingMeanReducer
from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.core.scnn.utils import Box
from lightstream.core.layers.streamingneighborhoodattention import StreamingNeighborhoodAttention2D


def test_rectangle_ownership_matches_dense_first_claim_with_tissue_mask():
    reducer = StreamingMeanReducer()
    reducer.start_stream(7, 9, 1, 1, torch.device("cpu"), torch.float64)
    boxes = [(0, 4, 0, 5), (0, 4, 3, 9), (2, 7, 0, 5), (2, 7, 3, 9)]
    tissue = torch.arange(63).reshape(7, 9) % 4 != 0
    dense_seen = torch.zeros((7, 9), dtype=torch.bool)

    for y0, y1, x0, x1 in boxes:
        expected = ~dense_seen[y0:y1, x0:x1] & tissue[y0:y1, x0:x1]
        actual = reducer._claim_region(
            (y0, y1, x0, x1), tissue[y0:y1, x0:x1],
            reducer._forward_claimed_boxes, torch.device("cpu"),
        )
        assert torch.equal(actual, expected)
        dense_seen[y0:y1, x0:x1] = True

    assert reducer._stream_seen_mask.numel() == 0
    with pytest.raises(RuntimeError, match="start_backward_replay"):
        reducer.claim_backward_region(boxes[0])
    reducer.start_backward_replay()
    dense_seen.zero_()
    for y0, y1, x0, x1 in boxes:
        expected = ~dense_seen[y0:y1, x0:x1] & tissue[y0:y1, x0:x1]
        actual = reducer.claim_backward_region((y0, y1, x0, x1), tissue[y0:y1, x0:x1])
        assert torch.equal(actual, expected)
        dense_seen[y0:y1, x0:x1] = True
    assert reducer._backward_seen_mask is None
    assert reducer._backward_replay_regions == []


def test_prepared_masks_are_shared_by_equivalent_heads():
    scnn = StreamingCNN.__new__(StreamingCNN)
    torch.nn.Module.__init__(scnn)
    first, second = StreamingMeanReducer(mask_resize=True), StreamingMeanReducer(mask_resize=True)
    scnn._reducer_head_map = {0: first, 1: second}
    scnn._current_output_heights = [5, 5]
    scnn._current_output_widths = [7, 7]
    scnn.device = torch.device("cpu")
    scnn._active_reducer_mask_image = torch.empty((1, 3, 10, 14))
    scnn._active_reducer_mask = torch.ones((1, 1, 10, 14), dtype=torch.bool)
    scnn._normalized_reducer_mask = None
    scnn._prepared_reducer_domain_masks = {}

    assert scnn._get_prepared_reducer_domain_mask(0) is scnn._get_prepared_reducer_domain_mask(1)
    assert len(scnn._prepared_reducer_domain_masks) == 1


def test_tile_reduction_saves_boolean_mask_for_float64_backward():
    value = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4).requires_grad_()
    valid = torch.tensor([[True, False, True, True], [False, True, True, False], [True, True, False, True]])
    saved = []

    def pack(tensor):
        saved.append(tensor.dtype)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        streaming_reduce_tile(value, valid, None).backward()

    assert torch.bool in saved
    assert torch.float64 not in saved
    torch.testing.assert_close(value.grad, valid[None, None].to(torch.float64))


def test_attention_replay_combines_input_and_parameter_gradients():
    class PointwiseAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.7, -1.2], dtype=torch.float64))
            self.kernel_size = 3

        def forward(self, x):
            return x * self.weight

    reference = PointwiseAttention()
    streamed = StreamingNeighborhoodAttention2D(attention=PointwiseAttention())
    streamed.input_loc = Box(0, 3, 0, 4, None)
    x = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    expected_input = x.clone().requires_grad_()
    actual_input = x.clone().requires_grad_()
    upstream = torch.randn_like(x)

    reference(expected_input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).backward(upstream)
    streamed(actual_input).backward(upstream)

    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    torch.testing.assert_close(streamed.attention.weight.grad, reference.weight.grad)
