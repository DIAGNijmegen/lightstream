import pytest
import torch

pytest.importorskip("numpy")

from lightstream.core.scnn.scnn import StreamingCNN
from lightstream.models.sshr.model import SSHRDecoder


def test_sshr_decoder_convs_retain_their_rectification_input_support():
    decoder = SSHRDecoder(
        encoder_channels=[8, 8, 8, 8],
        encoder_strides=[4, 8, 16, 32],
        kernel_size=1,
    )
    for parameter in decoder.parameters():
        torch.nn.init.constant_(parameter, 1.0)

    collector = StreamingCNN.__new__(StreamingCNN)
    torch.nn.Module.__init__(collector)
    collector.stream_module = decoder
    collector.dtype = torch.float32
    collector.device = torch.device("cpu")
    collector.eps = 1e-5
    collector.verbose = False
    collector._hooks = []
    collector._module_stats = {}
    collector._saved_tensors = {}
    collector._incoming_module_lost = {}
    collector._add_hooks_for_statistics()

    features = []
    for size in (16, 8, 4, 2):
        feature = torch.zeros(1, 8, size, size)
        feature[:, :, 1:, 1:] = 1
        features.append(feature)

    with torch.no_grad():
        decoder(features)

    for block, conv in zip(decoder.blocks, decoder.convs[:3]):
        assert collector._module_stats[conv]["input_lost"] == collector._module_stats[
            block.output_probe
        ]["lost"]

    collector._remove_hooks()
