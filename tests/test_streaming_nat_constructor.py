import copy

import pytest
import torch
from torch import nn

pytest.importorskip("timm")
pytest.importorskip("natten")

from lightstream.core.layers import (
    NeighborhoodAttention2D,
    StreamingNeighborhoodAttention2D,
)
from lightstream.models.nat import streaming as streaming_nat


class _MinimalNattenAttention(nn.Module):
    """NHWC attention-shaped backend with NATTEN's projection attributes."""

    kernel_size = 3
    dilation = 1

    def __init__(self, channels: int = 3):
        super().__init__()
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query, _, _ = self.qkv(inputs).chunk(3, dim=-1)
        return self.proj(query)


def test_streaming_nat_fresh_statistics_preserve_natten_projections(monkeypatch):
    monkeypatch.setattr(nn.Module, "cuda", lambda self, *args, **kwargs: self)
    source = nn.Sequential(NeighborhoodAttention2D(attention=_MinimalNattenAttention()))
    original_attention = source[0].attention
    original_qkv_type = type(original_attention.qkv)
    original_proj_type = type(original_attention.proj)
    original_qkv_shapes = [
        parameter.shape for parameter in original_attention.qkv.parameters()
    ]
    original_proj_shapes = [
        parameter.shape for parameter in original_attention.proj.parameters()
    ]

    monkeypatch.setattr(
        streaming_nat,
        "_VARIANTS",
        {
            "minimal": {
                "reference": {},
                "nchw": lambda: copy.deepcopy(source),
                "checkpoint": "unused",
            }
        },
    )
    monkeypatch.setattr(streaming_nat, "NAT", lambda **kwargs: copy.deepcopy(source))

    model = streaming_nat.StreamingNAT(
        "minimal",
        tile_size=8,
        pretrained=False,
        tile_cache={},
        verbose=False,
    )

    streamed_attention = model.stream_network.stream_module[0]
    assert model.stream_network.get_tile_cache()["net_stats"]
    assert isinstance(streamed_attention, StreamingNeighborhoodAttention2D)
    assert NeighborhoodAttention2D in model.constructor.keep_modules
    assert nn.Linear not in model.constructor.keep_modules

    qkv = streamed_attention.attention.qkv
    proj = streamed_attention.attention.proj
    assert type(qkv) is original_qkv_type
    assert type(proj) is original_proj_type
    assert [parameter.shape for parameter in qkv.parameters()] == original_qkv_shapes
    assert [parameter.shape for parameter in proj.parameters()] == original_proj_shapes


def test_prepared_streaming_nat_exposes_converted_backbone(monkeypatch):
    monkeypatch.setattr(nn.Module, "cuda", lambda self, *args, **kwargs: self)
    source = nn.Sequential(NeighborhoodAttention2D(attention=_MinimalNattenAttention()))
    supplied_state = copy.deepcopy(source.state_dict())

    monkeypatch.setattr(
        streaming_nat,
        "_VARIANTS",
        {
            "minimal": {
                "reference": {},
                "nchw": lambda: copy.deepcopy(source),
                "checkpoint": "unused",
            }
        },
    )
    monkeypatch.setattr(streaming_nat, "NAT", lambda **kwargs: copy.deepcopy(source))

    stream = streaming_nat.StreamingNAT(
        "minimal",
        tile_size=8,
        pretrained=supplied_state,
        tile_cache={},
        verbose=False,
    )

    assert stream._is_prepared
    converted_backbone = stream.stream_network.stream_module
    assert isinstance(converted_backbone, nn.Sequential)
    assert isinstance(converted_backbone[0], StreamingNeighborhoodAttention2D)
    assert torch.equal(
        converted_backbone[0].attention.qkv.weight,
        supplied_state["0.attention.qkv.weight"],
    )
