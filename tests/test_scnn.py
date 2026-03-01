import sys
import types
from pathlib import Path

import pytest
import torch

if "lightstream" not in sys.modules:
    pkg = types.ModuleType("lightstream")
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "lightstream")]
    sys.modules["lightstream"] = pkg

from lightstream.core.scnn.scnn import StreamingCNN


def _build_module():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=True),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 5, kernel_size=3, stride=1, padding=1, bias=True),
    ).eval()


@pytest.mark.parametrize("reduction_mode", ["none", "sum", "mean"])
def test_streaming_forward_reduction_modes_match_non_streaming(reduction_mode):
    torch.manual_seed(1)
    module = _build_module()
    image = torch.randn(1, 3, 25, 31)

    scnn = StreamingCNN(
        module,
        tile_shape=(1, 3, 16, 16),
        copy_to_gpu=False,
        statistics_on_cpu=True,
        dtype=torch.float32,
    )

    baseline = module(image)
    if reduction_mode == "none":
        expected = baseline
    elif reduction_mode == "sum":
        expected = baseline.sum(dim=(2, 3), keepdim=True)
    else:
        expected = baseline.mean(dim=(2, 3), keepdim=True)

    actual = scnn.forward(image, reduction_mode=reduction_mode)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_streaming_forward_invalid_reduction_mode_raises_value_error():
    module = _build_module()
    image = torch.randn(1, 3, 25, 31)

    scnn = StreamingCNN(
        module,
        tile_shape=(1, 3, 16, 16),
        copy_to_gpu=False,
        statistics_on_cpu=True,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="Unsupported reduction_mode"):
        scnn.forward(image, reduction_mode="median")
