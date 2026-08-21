from types import SimpleNamespace

import pytest
import torch

from lightstream.core.engine.operators import STREAMING_OPERATORS
from lightstream.core.engine.planner import (
    StreamingPlanBuilder,
    UnsupportedStreamingOperatorError,
)
from lightstream.core.layers import ChannelLayerNorm, StreamingConv2d


def test_registry_describes_pytorch_and_lightstream_capabilities():
    conv = STREAMING_OPERATORS.capabilities_for(torch.nn.Conv2d(2, 3, 3))
    assert conv.conversion and conv.statistics_forward and conv.statistics_backward
    assert conv.alignment and not conv.spatial_preserving

    norm = STREAMING_OPERATORS.capabilities_for(ChannelLayerNorm(3))
    assert norm.conversion and norm.spatial_preserving
    assert norm.statistics_forward and norm.statistics_backward

    streaming = STREAMING_OPERATORS.capabilities_for(StreamingConv2d(2, 3, 3))
    assert streaming.backward_tile_state


def test_plan_validation_reports_path_and_missing_capability():
    class Unknown(torch.nn.Module):
        def forward(self, value):
            return value

    facade = SimpleNamespace(stream_module=torch.nn.Sequential(torch.nn.Identity(), Unknown()))
    with pytest.raises(UnsupportedStreamingOperatorError) as error:
        StreamingPlanBuilder(facade).build(probe=False)

    assert "'1'" in str(error.value)
    assert "missing capability 'conversion'" in str(error.value)
