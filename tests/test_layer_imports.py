"""Static compatibility tests for the layer package migration."""

from lightstream.core import layers
from lightstream.core import scnn
from lightstream.core.layers.statisticsprobe import StatisticsProbe
from lightstream.core.layers.streamingconv import StreamingConv2d
from lightstream.core.layers.streaminglayernorm import ChannelLayerNorm, StreamingChannelLayerNorm
from lightstream.core.layers.streaminglayerscale import LayerScale, StreamingLayerScale
from lightstream.core.layers.streamingmerge import StreamingMerge
from lightstream.core.layers.streamingupsample import StreamingUpsample2d
from lightstream.core.scnn.statisticsprobe import StatisticsProbe as OldStatisticsProbe
from lightstream.core.scnn.streamingconv import StreamingConv2d as OldStreamingConv2d
from lightstream.core.scnn.streaminglayernorm import ChannelLayerNorm as OldChannelLayerNorm
from lightstream.core.scnn.streaminglayernorm import StreamingChannelLayerNorm as OldStreamingChannelLayerNorm
from lightstream.core.scnn.streaminglayerscale import LayerScale as OldLayerScale
from lightstream.core.scnn.streaminglayerscale import StreamingLayerScale as OldStreamingLayerScale
from lightstream.core.scnn.streamingmerge import StreamingMerge as OldStreamingMerge
from lightstream.core.scnn.streamingupsample import StreamingUpsample2d as OldStreamingUpsample2d


def test_implementation_module_compatibility_exports_are_identical():
    pairs = (
        (OldStatisticsProbe, StatisticsProbe),
        (OldStreamingConv2d, StreamingConv2d),
        (OldChannelLayerNorm, ChannelLayerNorm),
        (OldStreamingChannelLayerNorm, StreamingChannelLayerNorm),
        (OldLayerScale, LayerScale),
        (OldStreamingLayerScale, StreamingLayerScale),
        (OldStreamingMerge, StreamingMerge),
        (OldStreamingUpsample2d, StreamingUpsample2d),
    )
    assert all(old is new for old, new in pairs)


def test_public_package_compatibility_exports_are_identical():
    for name in scnn.__all__:
        assert getattr(scnn, name) is getattr(layers, name)
