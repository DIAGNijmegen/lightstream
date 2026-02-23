import pytest
import torch

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.scnn.streamingupsample import StreamingUpsample2d


def test_streaming_upsample_from_torch_matches_interpolate():
    upsample = torch.nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
    module = StreamingUpsample2d.from_torch_upsample(upsample)

    x = torch.rand(1, 3, 9, 11)
    expected = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    out = module(x)

    torch.testing.assert_close(out, expected)


def test_streaming_upsample_rejects_align_corners_true_for_bilinear():
    with pytest.raises(ValueError):
        StreamingUpsample2d(scale_factor=2.0, mode="bilinear", align_corners=True)


def test_constructor_keeps_upsample_modules():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.Upsample(scale_factor=2.0, mode="nearest"))
    constructor = StreamingConstructor(model, tile_size=32, verbose=False, statistics_on_cpu=True)
    assert torch.nn.Upsample in constructor.keep_modules
