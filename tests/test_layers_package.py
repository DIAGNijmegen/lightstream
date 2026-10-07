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


SUPPORTED_LAYER_CLASSES = {
    "ChannelLayerNorm": ChannelLayerNorm,
    "LayerScale": LayerScale,
    "StatisticsProbe": StatisticsProbe,
    "StreamingChannelLayerNorm": StreamingChannelLayerNorm,
    "StreamingConv2d": StreamingConv2d,
    "StreamingLayerScale": StreamingLayerScale,
    "StreamingMerge": StreamingMerge,
    "StreamingUpsample2d": StreamingUpsample2d,
}

DELETED_LAYER_MODULES = {
    "statisticsprobe",
    "streamingconv",
    "streaminglayernorm",
    "streaminglayerscale",
    "streamingmerge",
    "streamingupsample",
}


def test_layers_package_exports_supported_classes():
    assert {name: cls.__name__ for name, cls in SUPPORTED_LAYER_CLASSES.items()} == {
        name: name for name in SUPPORTED_LAYER_CLASSES
    }
    assert all(
        cls.__module__.startswith("lightstream.core.layers.")
        for cls in SUPPORTED_LAYER_CLASSES.values()
    )


def test_deleted_scnn_layer_module_paths_are_not_used():
    repository = Path(__file__).resolve().parents[1]
    forbidden_paths = {
        "lightstream.core.scnn." + module_name
        for module_name in DELETED_LAYER_MODULES
    }

    offenders = []
    for path in repository.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden_path in forbidden_paths:
            if forbidden_path in source:
                offenders.append(f"{path.relative_to(repository)}: {forbidden_path}")

    assert offenders == []


def test_legacy_scnn_layer_modules_were_deleted():
    scnn_package = Path(__file__).resolve().parents[1] / "lightstream" / "core" / "scnn"

    assert not {
        path.name
        for module_name in DELETED_LAYER_MODULES
        if (path := scnn_package / f"{module_name}.py").exists()
    }
