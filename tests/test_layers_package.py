import importlib

import pytest

from lightstream.core import layers


EXPECTED_LAYER_EXPORTS = {
    "ChannelLayerNorm",
    "LayerScale",
    "StatisticsProbe",
    "StreamingChannelLayerNorm",
    "StreamingConv2d",
    "StreamingLayerScale",
    "StreamingMerge",
    "StreamingUpsample2d",
}


def test_layers_package_exports_public_layer_classes():
    assert set(layers.__all__) == EXPECTED_LAYER_EXPORTS
    assert all(
        getattr(layers, name).__module__.startswith("lightstream.core.layers.")
        for name in layers.__all__
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "statisticsprobe",
        "streamingconv",
        "streaminglayernorm",
        "streaminglayerscale",
        "streamingmerge",
        "streamingupsample",
    ],
)
def test_removed_scnn_layer_modules_are_not_importable(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"lightstream.core.scnn.{module_name}")
