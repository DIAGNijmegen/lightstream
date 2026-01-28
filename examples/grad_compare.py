from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
import argparse

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


def _loss_grads_from_outputs(
    outputs: tuple[torch.Tensor, ...], target_value: float
) -> tuple[torch.Tensor, ...]:
    grads = []
    for out in outputs:
        target = torch.full_like(out, fill_value=target_value)
        grads.append(2 * (out - target) / out.numel())
    return tuple(grads)


def _compare_grads(stream_grads: dict[str, torch.Tensor], normal_grads: dict[str, torch.Tensor]) -> None:
    shared = sorted(set(stream_grads.keys()) & set(normal_grads.keys()))
    if not shared:
        print("No overlapping gradients found to compare.")
        return

    print(f"Comparing {len(shared)} parameter gradients:")
    for name in shared:
        diff = (stream_grads[name] - normal_grads[name]).abs()
        denom = normal_grads[name].abs().clamp_min(1e-12)
        rel = diff / denom
        print(
            f"{name}: "
            f"mean abs diff={diff.mean().item():.6e}, max abs diff={diff.max().item():.6e}, "
            f"mean rel diff={rel.mean().item():.6e}, max rel diff={rel.max().item():.6e}"
        )


def _compare_conv_weight_grads(
    stream_grads: dict[str, torch.Tensor], normal_grads: dict[str, torch.Tensor]
) -> None:
    conv_names = sorted(
        name
        for name, grad in stream_grads.items()
        if name in normal_grads and grad.ndim == 4 and "conv" in name
    )
    if not conv_names:
        print("No convolution weight gradients found to compare.")
        return

    print("\nConvolution kernel gradient stats:")
    for name in conv_names:
        stream_grad = stream_grads[name]
        normal_grad = normal_grads[name]
        diff = (stream_grad - normal_grad).abs()
        print(
            f"{name}: "
            f"stream mean abs={stream_grad.abs().mean().item():.6e}, "
            f"normal mean abs={normal_grad.abs().mean().item():.6e}, "
            f"max abs diff={diff.max().item():.6e}"
        )


def _parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    key = value.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{value}'. Choose from: {', '.join(mapping.keys())}")
    return mapping[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare streaming vs non-streaming backward gradients for WSS.")
    parser.add_argument("--dtype", default="float64", help="float16, float32, or float64")
    parser.add_argument("--tile-size", type=int, default=2560)
    parser.add_argument("--input-size", type=int, default=3200)
    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _parse_dtype(args.dtype)
    tile_size = args.tile_size
    input_size = args.input_size

    img = torch.rand((1, 3, input_size, input_size), device=device, dtype=dtype)

    network = StreamingWSS(
        "resnet18",
        tile_size,
        additional_modules=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
        saliency=True,
    ).to(device=device, dtype=dtype)
    network.stream_network.device = device
    network.stream_network.mean = network.stream_network.mean.to(device=device, dtype=dtype)
    network.stream_network.std = network.stream_network.std.to(device=device, dtype=dtype)

    _zero_grads(network.stream_network.stream_module.parameters())
    stream_outputs = network(img)
    target_value = 50.0
    stream_grads = _loss_grads_from_outputs(stream_outputs, target_value)
    network.stream_network.backward(img, stream_grads)
    streaming_param_grads = _gather_param_grads(network.stream_network.stream_module)

    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    _zero_grads(normal_net.parameters())
    img_normal = img.detach().clone().requires_grad_(True)
    normal_outputs = normal_net(img_normal)
    forward_diffs = []
    for stream_out, normal_out in zip(stream_outputs, normal_outputs):
        forward_diffs.append((stream_out - normal_out).sum().item())
    print(f"Forward output sum diffs: {forward_diffs}")

    normal_loss = sum(torch.nn.functional.mse_loss(out, torch.full_like(out, target_value)) for out in normal_outputs)
    normal_loss.backward()
    normal_param_grads = _gather_param_grads(normal_net)

    if img_normal.grad is not None:
        input_grad_diff = img_normal.grad.detach().cpu().numpy() - network.stream_network.saliency_map[0].numpy()
        print(f"Input gradient max diff: {input_grad_diff.max()}")

    _compare_grads(streaming_param_grads, normal_param_grads)
    _compare_conv_weight_grads(streaming_param_grads, normal_param_grads)


if __name__ == "__main__":
    main()
