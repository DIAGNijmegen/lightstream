import pytest
import torch

from lightstream.core.reducer import AttentionGeMReducer, FusedAttentionGeMReducer, GeMReducer, MeanReducer, SumReducer
from lightstream.core.reducer.utils import prepare_spatial_mask


def test_prepare_spatial_mask_rejects_mismatch_without_resize():
    x = torch.zeros(2, 3, 4, 6)
    mask = torch.ones(2, 2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="mask spatial shape .* set mask_resize=True"):
        prepare_spatial_mask(mask, x)


def test_prepare_spatial_mask_resizes_to_bool_n1hw_on_input_device():
    x = torch.zeros(2, 3, 4, 6)
    mask = torch.tensor(
        [
            [[True, False, True], [False, True, False]],
            [[False, True, False], [True, False, True]],
        ]
    )

    prepared = prepare_spatial_mask(mask, x, mask_resize=True)

    assert prepared.shape == (2, 1, 4, 6)
    assert prepared.dtype == torch.bool
    assert prepared.device == x.device
    assert torch.equal(prepared[..., 0:2, 0:2], mask[:, None, 0:1, 0:1].expand(-1, -1, 2, 2))


def test_prepare_spatial_mask_rejects_non_nearest_resize_mode():
    x = torch.zeros(1, 1, 4, 4)
    mask = torch.ones(2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="Only 'nearest' is supported"):
        prepare_spatial_mask(mask, x, mask_resize=True, mask_resize_mode="bilinear")


@pytest.mark.parametrize("reducer_cls", [MeanReducer, SumReducer, GeMReducer])
def test_single_input_reducers_resize_masks_when_enabled(reducer_cls):
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) + 1
    mask = torch.tensor([[True, False], [False, True]])
    expected_mask = prepare_spatial_mask(mask, x, mask_resize=True)
    expected = reducer_cls()(x, mask=expected_mask)

    assert torch.allclose(reducer_cls(mask_resize=True)(x, mask=mask), expected)


def test_attention_gem_resizes_mask_when_enabled():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) + 1
    logits = torch.zeros(1, 1, 4, 4)
    mask = torch.tensor([[True, False], [False, True]])
    expected_mask = prepare_spatial_mask(mask, x, mask_resize=True)
    reducer = AttentionGeMReducer(r_init=1.0)

    assert torch.allclose(
        AttentionGeMReducer(r_init=1.0, mask_resize=True)(x, logits, mask=mask),
        reducer(x, logits, mask=expected_mask),
    )


def test_fused_attention_gem_resizes_mask_against_y1_domain():
    y1 = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) + 1
    y2 = y1 + 1
    y3 = y1 + 2
    logits = tuple(torch.zeros(1, 1, 4, 4) for _ in range(3))
    mask = torch.tensor([[True, False], [False, True]])
    expected_mask = prepare_spatial_mask(mask, y1, mask_resize=True)
    reducer = FusedAttentionGeMReducer(r_init=1.0)

    assert torch.allclose(
        FusedAttentionGeMReducer(r_init=1.0, mask_resize=True)(y1, y2, y3, *logits, mask=mask),
        reducer(y1, y2, y3, *logits, mask=expected_mask),
    )
