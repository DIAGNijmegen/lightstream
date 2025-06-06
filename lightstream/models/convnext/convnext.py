from pathlib import Path
from typing import Callable

import torch.nn as nn
from torch.nn import Sequential

from lightstream.modules.lightningstreaming import StreamingModule

from torchvision.ops import StochasticDepth
from torchvision.models.convnext import convnext_tiny, convnext_small


def _toggle_stochastic_depth(model, training=False):
    for m in model.modules():
        if isinstance(m, StochasticDepth):
            m.training = training


def _set_layer_scale(model, val=1.0):
    for x in model.modules():
        if hasattr(x, "layer_scale"):
            x.layer_scale.data.fill_(val)


class StreamingConvNext(StreamingModule):
    def __init__(
        self,
        encoder: str,
        tile_size: int,
        additional_modules: nn.Module | None = None,
        remove_last_block=False,
        use_stochastic_depth: bool = False,
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
        model_choices = {"convnext-tiny": convnext_tiny, "convnext-small": convnext_small}
        self.use_stochastic_depth = use_stochastic_depth

        if encoder not in model_choices:
            raise ValueError(f"Invalid model name '{encoder}'. " f"Choose one of: {', '.join(model_choices.keys())}")

        stream_network = model_choices[encoder](weights="DEFAULT").features

        if remove_last_block:
            stream_network = stream_network[0:6]

        if additional_modules is not None:
            stream_network = Sequential(stream_network, additional_modules)

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
            before_streaming_init_callbacks=[_set_layer_scale],
            after_streaming_init_callbacks=[_toggle_stochastic_depth],
        )
        _toggle_stochastic_depth(self.stream_network.stream_module, training=self.use_stochastic_depth)

    @staticmethod
    def get_model_choices() -> dict[str, Callable[..., nn.Module]]:
        return {"convnext-tiny": convnext_tiny, "convnext-small": convnext_small}

    @classmethod
    def get_model_names(cls) -> list[str]:
        return list(cls.get_model_choices().keys())


if __name__ == "__main__":
    import torch

    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 6400, 6400)).to("cuda")
    network = StreamingConvNext(
        "convnext-tiny",
        6400,
        additional_modules=torch.nn.MaxPool2d(8, 8),
        remove_last_block=False,
        use_stochastic_depth=False,
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
