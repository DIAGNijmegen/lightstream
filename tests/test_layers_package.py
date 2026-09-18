from pathlib import Path

from lightstream.core.layers import (
    ChannelLayerNorm,
    LayerScale,
    StatisticsProbe,
    StreamingChannelLayerNorm,
    StreamingConv2d,
    StreamingLayerScale,
    StreamingMerge,
    StreamingUpsample2d,
)


def test_layers_package_exports_supported_classes():
    exported_classes = {
        ChannelLayerNorm,
        LayerScale,
        StatisticsProbe,
        StreamingChannelLayerNorm,
        StreamingConv2d,
        StreamingLayerScale,
        StreamingMerge,
        StreamingUpsample2d,
    }

    assert {cls.__name__ for cls in exported_classes} == {
        "ChannelLayerNorm",
        "LayerScale",
        "StatisticsProbe",
        "StreamingChannelLayerNorm",
        "StreamingConv2d",
        "StreamingLayerScale",
        "StreamingMerge",
        "StreamingUpsample2d",
    }


def test_deleted_scnn_layer_module_paths_are_not_used():
    repository = Path(__file__).resolve().parents[1]
    deleted_modules = {
        "statisticsprobe",
        "streamingconv",
        "streaminglayernorm",
        "streaminglayerscale",
        "streamingmerge",
        "streamingupsample",
    }
    forbidden_paths = {
        "lightstream.core.scnn." + module_name for module_name in deleted_modules
    }

    offenders = []
    for path in repository.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden_path in forbidden_paths:
            if forbidden_path in source:
                offenders.append(f"{path.relative_to(repository)}: {forbidden_path}")

    assert offenders == []
