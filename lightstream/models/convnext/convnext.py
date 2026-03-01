from pathlib import Path

import torch.nn as nn
from torch.nn import Sequential
from typing import Callable

from timm.models.convnext import (
    convnext_atto,
    convnext_atto_ols,
    convnext_femto,
    convnext_femto_ols,
    convnext_pico,
    convnext_pico_ols,
    convnext_nano,
    convnext_nano_ols,
    convnext_tiny_hnf,
    convnext_tiny,
    convnext_small
)
from lightstream.modules.lightningstreaming import StreamingModule


def _set_layer_gamma(model, val=1.0):
    for x in model.modules():
        if hasattr(x, "gamma"):
            x.gamma.data.fill_(val)


class StreamingConvNext(StreamingModule):
    def __init__(
        self,
        encoder: str,
        tile_size: int,
        additional_modules: nn.Module | None = None,
        remove_last_block=False,
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

        network = model_choices[encoder](pretrained=True)

        end = 3 if remove_last_block else 4

        if additional_modules is not None:
            stream_network = Sequential(network.stem, network.stages[0:end], additional_modules)
        else:
            stream_network = Sequential(network.stem, network.stages[0:end])

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
            before_streaming_init_callbacks=[_set_layer_gamma],
        )

    @staticmethod
    def get_model_choices() -> dict[str, Callable[..., nn.Module]]:
        return {
            "convnext_atto": convnext_atto,
            "convnext_atto_ols": convnext_atto_ols,
            "convnext_femto": convnext_femto,
            "convnext_femto_ols": convnext_femto_ols,
            "convnext_pico": convnext_pico,
            "convnext_pico_ols": convnext_pico_ols,
            "convnext_nano": convnext_nano,
            "convnext_nano_ols": convnext_nano_ols,
            "convnext_tiny_hnf": convnext_tiny_hnf,
            "convnext_tiny": convnext_tiny,
            "convnext_small": convnext_small
        }

    @classmethod
    def get_model_names(cls) -> list[str]:
        return list(cls.get_model_choices().keys())


if __name__ == "__main__":
    import torch

    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 5440, 5440)).to("cuda")
    network = StreamingConvNext(
        "convnext_tiny_hnf",
        4800,
        additional_modules=None,
        remove_last_block=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
    )
    network.to("cuda")
    network.stream_network.device = torch.device("cuda")

    network.stream_network.mean = network.stream_network.mean.to("cuda")
    network.stream_network.std = network.stream_network.std.to("cuda")

    out_streaming = network(img)
    print(network.tile_size)

    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    out_normal = normal_net(img)
    diff = out_streaming - out_normal
    print(diff.max())
