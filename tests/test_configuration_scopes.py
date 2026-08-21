"""Failure-injection tests for setup-only configuration state."""

import pytest
import torch
from torch import nn

from lightstream.core.engine.configuration import (
    cudnn_flags,
    gradient_mode,
    normalization_passthrough,
    reducer_passthrough,
    statistics_hooks,
    temporary_parameters,
)
from lightstream.core.layers.streaminglayernorm import ChannelLayerNorm
from lightstream.core.reducer import MeanReducer


class _Runtime:
    def __init__(self):
        self.stream_module = nn.Sequential(nn.Conv2d(1, 1, 1), ChannelLayerNorm(1), MeanReducer())
        self._hooks = []

    def _add_hooks_for_statistics(self):
        self._hooks.append(self.stream_module[0].register_forward_hook(lambda *args: None))

    def _remove_hooks(self):
        hooks, self._hooks = self._hooks, []
        for hook in hooks:
            hook.remove()

    def _reset_parameters_to_constant(self):
        self.stream_module[0].weight.data.fill_(7)


def test_configuration_scopes_restore_state_when_probe_raises():
    runtime = _Runtime()
    original_weight = runtime.stream_module[0].weight.detach().clone()
    reducer = runtime.stream_module[2]
    norm = runtime.stream_module[1]
    reducer._streaming_passthrough = False
    norm._streaming_statistics_passthrough = False
    original_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    original_grad_mode = torch.is_grad_enabled()

    with pytest.raises(RuntimeError, match="injected"):
        with (
            cudnn_flags(deterministic=True, benchmark=False),
            temporary_parameters(runtime),
            statistics_hooks(runtime),
            reducer_passthrough(runtime),
            normalization_passthrough(runtime),
            gradient_mode(not original_grad_mode),
        ):
            assert runtime._hooks
            raise RuntimeError("injected configuration failure")

    torch.testing.assert_close(runtime.stream_module[0].weight, original_weight)
    assert runtime._hooks == []
    assert reducer._streaming_passthrough is False
    assert norm._streaming_statistics_passthrough is False
    assert torch.is_grad_enabled() is original_grad_mode
    assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) == original_flags
