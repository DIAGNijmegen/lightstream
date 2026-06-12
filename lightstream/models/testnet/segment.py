from pathlib import Path

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Sequential

from lightstream.modules.streaming import StreamingModule
from lightstream.core.scnn.streaminglayernorm import ChannelLayerNorm


class StreamingTestNet(StreamingModule):
    def __init__(
        self,
        tile_size: int,
        verbose: bool = True,
        deterministic: bool = True,
        saliency: bool = False,
        copy_to_gpu: bool = False,
        statistics_on_cpu: bool = True,
        normalize_on_gpu: bool = True,
        mean: list | None = None,
        std: list | None = None,
        tile_cache_path=None,
    ):
        stream_network = self.create_model()

        if mean is None:
            mean = [0, 0, 0]
        if std is None:
            std = [1, 1, 1]

        if tile_cache_path is None:
            tile_cache_path = Path.cwd() / Path(
                f"testnet_tile_cache_1_3_{str(tile_size)}_{str(tile_size)}"
            )

        super().__init__(
            stream_network,
            tile_size,
            tile_cache_path,
            verbose=verbose,
            deterministic=deterministic,
            saliency=saliency,
            copy_to_gpu=copy_to_gpu,
            statistics_on_cpu=statistics_on_cpu,
            normalize_on_gpu=normalize_on_gpu,
            mean=mean,
            std=std,
            add_keep_modules=[nn.BatchNorm2d],
        )

    @staticmethod
    def create_model():
        padding = 1

        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            ChannelLayerNorm(16),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            ChannelLayerNorm(32),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            ChannelLayerNorm(64),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            ChannelLayerNorm(128),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            ChannelLayerNorm(256),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            ChannelLayerNorm(256),
            torch.nn.MaxPool2d(2),
        )


        decoder = torch.nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=padding),
            torch.nn.ReLU(),
        )

        classifier = torch.nn.Conv2d(32, 1, kernel_size=3, padding=1)

        stream_net = torch.nn.Sequential(encoder, decoder, classifier)

        return stream_net


if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    dtype = torch.float64
    img = torch.rand((1, 3, 4800, 4800)).to("cuda", dtype=dtype)
    network = StreamingTestNet(
        1920,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
    )
    network.to("cuda", dtype=dtype)
    network.stream_network.device = torch.device("cuda")

    network.stream_network.mean = network.stream_network.mean.to("cuda", dtype=dtype)
    network.stream_network.std = network.stream_network.std.to("cuda", dtype=dtype)

    out_streaming = network(img)
    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    out_normal = normal_net(img)
    diff = out_streaming - out_normal
    print(diff.max())
