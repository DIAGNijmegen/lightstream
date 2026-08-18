import torch

from lightstream.core.engine.stitching import stitch_window


def test_stitch_window_without_a_network():
    output = torch.zeros(1, 1, 4, 5)
    tile = torch.arange(12).reshape(1, 1, 3, 4)
    stitch_window(output, tile, (1, 3, 2, 5), (0, 2, 1, 4))
    assert torch.equal(output[:, :, 1:3, 2:5], tile[:, :, 0:2, 1:4])
