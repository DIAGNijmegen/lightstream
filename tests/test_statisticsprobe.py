import operator

import pytest
import torch

from lightstream.core.scnn.statisticsprobe import StatisticsProbe


def _valid(tensor: torch.Tensor) -> torch.Tensor:
    # Streaming statistics use the maximum positive probe value to identify
    # positions to which every required branch contributed.
    return tensor == tensor.max()


@pytest.mark.parametrize("operation", [operator.mul, operator.add])
@pytest.mark.parametrize("reversed_operands", [False, True])
def test_probe_arithmetic_support_and_branch_gradients(operation, reversed_operands):
    """Arithmetic support is its operands' intersection in statistics probes."""
    shallow = torch.zeros(1, 1, 6, 7)
    shallow[:, :, 1:6, 1:6] = 2.0
    weights = torch.zeros_like(shallow)
    weights[:, :, 0:5, 2:7] = 3.0
    shallow.requires_grad_()
    weights.requires_grad_()

    shallow_seen = StatisticsProbe()(shallow)
    weights_seen = StatisticsProbe()(weights)
    operands = (weights_seen, shallow_seen) if reversed_operands else (shallow_seen, weights_seen)
    weighted_features = StatisticsProbe()(operation(*operands))
    output = StatisticsProbe()(weighted_features)

    expected_valid = _valid(shallow_seen) & _valid(weights_seen)
    assert torch.equal(_valid(weighted_features), expected_valid)
    assert torch.equal(_valid(output), expected_valid)

    # Positive, nonuniform upstream gradients make both branch derivatives
    # observable and prevent cancellation or a uniform reduction hiding one.
    upstream = torch.arange(1, output.numel() + 1, dtype=output.dtype).reshape_as(output)
    output.backward(upstream)
    if operation is operator.mul:
        assert torch.equal(shallow.grad, upstream * weights.detach())
        assert torch.equal(weights.grad, upstream * shallow.detach())
    else:
        assert torch.equal(shallow.grad, upstream)
        assert torch.equal(weights.grad, upstream)


def test_statistics_probe_is_an_identity_in_ordinary_execution():
    value = torch.randn(2, 3, 4, 5, requires_grad=True)
    observed = StatisticsProbe()(value)

    assert observed is value
