from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import Sequential

from torchvision.models import resnet18, resnet34, resnet50
from lightstream.modules.streaming import StreamingModule
from lightstream.models.segment.model import WSS


class StreamingWSS(StreamingModule):
    def __init__(
        self,
        encoder: str,
        tile_size: int,
        additional_modules: nn.Module | None = None,
        remove_last_block=True,
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
        model_choices = self.get_model_choices()

        if encoder not in model_choices:
            raise ValueError(f"Invalid model name '{encoder}'. " f"Choose one of: {', '.join(model_choices.keys())}")

        if additional_modules is not None:
            stream_network = Sequential(
                WSS(encoder=encoder, weights="default", remove_last_block=remove_last_block),
                additional_modules,
            )
        else:
            stream_network = WSS(encoder=encoder, weights="default", remove_last_block=remove_last_block)

        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        if tile_cache_path is None:
            tile_cache_path = Path.cwd() / Path(f"{encoder}_tile_cache_1_3_{str(tile_size)}_{str(tile_size)}")

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
    def get_model_choices() -> dict[str, Callable[..., nn.Module]]:
        return {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
        }

    @classmethod
    def get_model_names(cls) -> list[str]:
        return list(cls.get_model_choices().keys())


if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    dtype=torch.float64
    img = torch.rand((1, 3, 4800, 4800)).to("cuda", dtype=dtype)
    network = StreamingWSS(
        "resnet18",
        3200,
        additional_modules=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
    )
    network.to("cuda", dtype=dtype)
    network.eval()
    network.stream_network.device = torch.device("cuda")


    network.stream_network.mean = network.stream_network.mean.to("cuda", dtype=dtype)
    network.stream_network.std = network.stream_network.std.to("cuda", dtype=dtype)

    out_streaming = network(img)
    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    out_normal = normal_net(img)

    for stream_outs, normal_outs in zip(out_streaming, out_normal):
        diff = (stream_outs - normal_outs).abs()
        print(f"Forward output sum/max diff: {diff.sum().item()}, {diff.max().item()}")
