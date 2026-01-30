from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import Sequential

from lightstream.modules.streaming import StreamingModule


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
            tile_cache_path = Path.cwd() / Path(f"testnet_tile_cache_1_3_{str(tile_size)}_{str(tile_size)}")

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
        padding = 0

        stream_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=padding), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=padding), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=padding), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=padding), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=padding), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=padding), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))

        return stream_net

if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    dtype=torch.float32
    img = torch.rand((1, 3, 4800, 4800)).to("cuda", dtype=dtype)
    network = StreamingTestNet(
        3200,
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
    diff = (out_streaming - out_normal).abs()
    print(diff.max())

    print(f"Forward output sum/max diff: {diff.sum().item()}, {diff.max().item()}")