import sys
import types

import pytest
import torch

from lightstream.core.scnn import ChannelLayerNorm
from lightstream.core.scnn.streaminglayernorm import ChannelLayerNorm as ImportedChannelLayerNorm


def test_channel_layer_norm_matches_nhwc_layer_norm():
    torch.manual_seed(7)
    module = ChannelLayerNorm(3, eps=1e-6, elementwise_affine=True)
    reference = torch.nn.LayerNorm(3, eps=1e-6, elementwise_affine=True)
    reference.load_state_dict(module.norm.state_dict())

    x = torch.randn(2, 3, 5, 7)
    expected = reference(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    torch.testing.assert_close(module(x), expected)


def test_channel_layer_norm_rejects_non_4d_input():
    module = ChannelLayerNorm(3)

    with pytest.raises(ValueError, match="expects 4D NCHW input"):
        module(torch.randn(2, 3, 5))


def test_channel_layer_norm_rejects_channel_mismatch():
    module = ChannelLayerNorm(3)

    with pytest.raises(ValueError, match="expected 3 channels"):
        module(torch.randn(2, 4, 5, 7))


def test_channel_layer_norm_is_public_from_scnn_package():
    assert ImportedChannelLayerNorm is ChannelLayerNorm


def test_constructor_considers_only_channel_layer_norm_streamable(monkeypatch):
    # The constructor imports StreamingCNN, whose module imports numpy for full
    # streaming execution. This test only needs the constructor keep-list, so a
    # minimal import-time stub keeps the unit test focused when numpy is not
    # installed in the local environment.
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    from lightstream.core.constructor import StreamingConstructor

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 1),
        ChannelLayerNorm(3),
        torch.nn.LayerNorm(3),
    )
    constructor = StreamingConstructor(model, tile_size=32, verbose=False, statistics_on_cpu=True)

    assert ChannelLayerNorm in constructor.keep_modules
    assert torch.nn.LayerNorm not in constructor.keep_modules


def test_streaming_statistics_hooks_include_channel_layer_norm(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))

    from lightstream.core.scnn.scnn import StreamingCNN
    from lightstream.core.scnn.utils import Lost

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
        ChannelLayerNorm(3),
        torch.nn.Conv2d(3, 3, kernel_size=1, bias=False),
    ).eval()

    scnn = StreamingCNN(
        model,
        tile_shape=(1, 3, 6, 6),
        verbose=False,
        deterministic=True,
        copy_to_gpu=False,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
    )

    norm = scnn.stream_module[1]
    stats = scnn._module_stats[norm]

    assert stats["stride"].tolist() == [1, 1, 1]
    assert stats["lost"] == Lost(1, 1, 1, 1)
    assert stats["grad_lost"] == Lost(1, 1, 1, 1)
    assert stats["output_stride"].tolist() == [1, 1, 1]
