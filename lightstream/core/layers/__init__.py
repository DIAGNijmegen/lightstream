"""Public layers used to construct and execute streaming CNNs."""

from lightstream.core.layers.statisticsprobe import StatisticsProbe
from lightstream.core.layers.streamingconv import StreamingConv2d
from lightstream.core.layers.streaminglayernorm import ChannelLayerNorm, StreamingChannelLayerNorm
from lightstream.core.layers.streaminglayerscale import LayerScale, StreamingLayerScale
from lightstream.core.layers.streamingmerge import StreamingMerge
from lightstream.core.layers.streamingupsample import StreamingUpsample2d
from lightstream.core.engine.operators import register_operator

# Lightstream-owned layers publish their capabilities alongside their public
# imports, so callers do not need to import the legacy StreamingCNN facade.
for _pointwise in (ChannelLayerNorm, LayerScale, StatisticsProbe, StreamingMerge):
    register_operator(
        _pointwise, conversion=True, statistics_forward=True,
        statistics_backward=True, spatial_preserving=True,
    )
register_operator(
    StreamingConv2d, conversion=True, statistics_forward=True,
    statistics_backward=True, backward_tile_state=True, alignment=True,
    restore=lambda module, facade: module.to_torch_conv2d(),
)
register_operator(
    StreamingUpsample2d, conversion=True, statistics_forward=True,
    statistics_backward=True, backward_tile_state=True,
    restore=lambda module, facade: module.to_torch_upsample(),
)
register_operator(
    StreamingChannelLayerNorm, conversion=True, statistics_forward=True,
    statistics_backward=True, spatial_preserving=True, backward_tile_state=True,
    restore=lambda module, facade: module.to_channel_layer_norm(),
)
register_operator(
    StreamingLayerScale, conversion=True, statistics_forward=True,
    statistics_backward=True, spatial_preserving=True, backward_tile_state=True,
    restore=lambda module, facade: module.to_layer_scale(),
)

__all__ = [
    "ChannelLayerNorm",
    "LayerScale",
    "StatisticsProbe",
    "StreamingChannelLayerNorm",
    "StreamingConv2d",
    "StreamingLayerScale",
    "StreamingMerge",
    "StreamingUpsample2d",
]
