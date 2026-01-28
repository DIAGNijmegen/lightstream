from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

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
        remove_last_block: bool = True,
        verbose: bool = True,
        deterministic: bool = True,
        saliency: bool = False,
        copy_to_gpu: bool = False,
        statistics_on_cpu: bool = True,
        normalize_on_gpu: bool = True,
        mean: list | None = None,
        std: list | None = None,
        tile_cache_path: Path | None = None,
    ):
        model_choices = self.get_model_choices()

        if encoder not in model_choices:
            raise ValueError(f"Invalid model name '{encoder}'. Choose one of: {', '.join(model_choices.keys())}")

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


def _gather_param_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def _zero_grads(parameters: Iterable[torch.nn.Parameter]) -> None:
    for param in parameters:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def _loss_grads_from_outputs(outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    grads = []
    for out in outputs:
        grads.append(torch.ones_like(out) / out.numel())
    return tuple(grads)


def _compare_grads(stream_grads: dict[str, torch.Tensor], normal_grads: dict[str, torch.Tensor]) -> None:
    shared = sorted(set(stream_grads.keys()) & set(normal_grads.keys()))
    if not shared:
        print("No overlapping gradients found to compare.")
        return

    print(f"Comparing {len(shared)} parameter gradients:")
    for name in shared:
        diff = (stream_grads[name] - normal_grads[name]).abs()
        print(
            f"{name}: mean abs diff={diff.mean().item():.6e}, max abs diff={diff.max().item():.6e}"
        )


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_size = 2560
    input_size = 3200

    img = torch.rand((1, 3, input_size, input_size), device=device)

    network = StreamingWSS(
        "resnet18",
        tile_size,
        additional_modules=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
    ).to(device)
    network.stream_network.device = device
    network.stream_network.mean = network.stream_network.mean.to(device)
    network.stream_network.std = network.stream_network.std.to(device)

    _zero_grads(network.stream_network.stream_module.parameters())
    stream_outputs = network(img)
    stream_grads = _loss_grads_from_outputs(stream_outputs)
    network.stream_network.backward(img, stream_grads)
    streaming_param_grads = _gather_param_grads(network.stream_network.stream_module)

    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    _zero_grads(normal_net.parameters())
    normal_outputs = normal_net(img)
    normal_loss = sum(out.mean() for out in normal_outputs)
    normal_loss.backward()
    normal_param_grads = _gather_param_grads(normal_net)

    _compare_grads(streaming_param_grads, normal_param_grads)


if __name__ == "__main__":
    main()
