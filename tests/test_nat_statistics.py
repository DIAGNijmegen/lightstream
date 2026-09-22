import types

import pytest
import torch
from torch import nn

from lightstream.core.layers.streamingneighborhoodattention import (
    NeighborhoodAttention2D,
)
from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.core.scnn.utils import Lost


class _IdentityNeighborhoodAttention(nn.Module):
    kernel_size = (5, 3)
    dilation = (1, 2)

    def forward(self, inputs):
        return inputs


def _statistics_collector():
    collector = StreamingCNN.__new__(StreamingCNN)
    collector.eps = 1e-5
    collector.dtype = torch.float32
    collector.device = torch.device("cpu")
    collector._saved_tensors = {}
    collector._module_stats = {}
    collector._print_verbose = lambda *args, **kwargs: None
    return collector


@pytest.mark.parametrize("invalid_output_value", [0.0, float("nan")])
def test_nat_statistics_derive_validity_from_input_and_declared_support(
    invalid_output_value,
):
    collector = _statistics_collector()
    module = NeighborhoodAttention2D(attention=_IdentityNeighborhoodAttention())
    inputs = torch.zeros(1, 2, 12, 13)
    inputs[:, :, 1:-2, 2:-1] = 1
    output = torch.full_like(inputs, invalid_output_value)

    inspected_tensors = []
    original_border_amount = collector._non_max_border_amount

    def record_border_source(self, tensor):
        inspected_tensors.append(tensor)
        return original_border_amount(tensor)

    collector._non_max_border_amount = types.MethodType(
        record_border_source, collector
    )

    with torch.no_grad():
        collector._forward_gather_statistics_hook(module, (inputs,), output)

    expected_lost = Lost(top=3, left=4, bottom=4, right=3)
    assert len(inspected_tensors) == 1
    assert inspected_tensors[0] is inputs
    assert collector._module_stats[module]["lost"] == expected_lost

    expected_mask = torch.zeros_like(output)
    expected_mask[
        :,
        :,
        expected_lost.top : -expected_lost.bottom,
        expected_lost.left : -expected_lost.right,
    ] = 1
    torch.testing.assert_close(output, expected_mask)
