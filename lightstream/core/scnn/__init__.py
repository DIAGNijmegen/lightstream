"""Compatibility exports for streaming CNN layers.

New code should import layer building blocks from :mod:`lightstream.core.layers`.
The exports here remain aliases to preserve the existing public API.
"""

from lightstream.core.layers import (
    ChannelLayerNorm,
    LayerScale,
    StreamingChannelLayerNorm,
    StreamingLayerScale,
    StreamingMerge,
)

__all__ = ["ChannelLayerNorm", "StreamingChannelLayerNorm", "LayerScale", "StreamingLayerScale", "StreamingMerge"]
