from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch.nn import Sequential
from torchvision.models import resnet18, resnet34, resnet50

from lightstream.models.segment.model import WSS
from lightstream.modules.streaming import StreamingModule


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


def _freeze_batchnorm(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.BatchNorm2d):
            submodule.eval()
            for param in submodule.parameters():
                param.requires_grad = False


def _zero_grads(parameters: Iterable[torch.nn.Parameter]) -> None:
    for param in parameters:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def _gather_param_grads(model: nn.Module, reducer_only: bool = False) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if reducer_only and not any(tag in name for tag in ("decoder", "reducer")):
            continue
        grads[name] = param.grad.detach().clone()
    return grads


def _print_grad_stats(
    title: str,
    stream_grads: dict[str, torch.Tensor],
    normal_grads: dict[str, torch.Tensor],
) -> None:
    names = sorted(set(stream_grads) & set(normal_grads))
    if not names:
        print(f"{title}: no overlapping gradients found.")
        return

    print(f"\n{title} ({len(names)} params)")
    for name in names:
        diff = (stream_grads[name] - normal_grads[name]).abs()
        print(
            f"  {name}: "
            f"stream|g|={stream_grads[name].abs().mean().item():.6e}, "
            f"normal|g|={normal_grads[name].abs().mean().item():.6e}, "
            f"mean|Δg|={diff.mean().item():.6e}, "
            f"max|Δg|={diff.max().item():.6e}"
        )


def _print_head_forward_stats(stream_outputs: tuple[torch.Tensor, ...], normal_outputs: tuple[torch.Tensor, ...]) -> None:
    print("\nForward per-head diagnostics:")
    for idx, (stream_out, normal_out) in enumerate(zip(stream_outputs, normal_outputs)):
        diff = (stream_out - normal_out).abs()
        print(
            f"  head[{idx}]: "
            f"stream_mean={stream_out.mean().item():.6e}, "
            f"normal_mean={normal_out.mean().item():.6e}, "
            f"mean|Δ|={diff.mean().item():.6e}, "
            f"max|Δ|={diff.max().item():.6e}"
        )


def _print_reducer_backward_consistency(
    stream_map_grad: torch.Tensor,
    normal_map_grad: torch.Tensor,
) -> None:
    diff = (stream_map_grad - normal_map_grad).abs()
    print("\nReducer backward consistency (post-reduce(stream_map) dL/dstream_map):")
    print(
        "  stream_map_grad: "
        f"stream|g|={stream_map_grad.abs().mean().item():.6e}, "
        f"normal|g|={normal_map_grad.abs().mean().item():.6e}, "
        f"mean|Δg|={diff.mean().item():.6e}, "
        f"max|Δg|={diff.max().item():.6e}"
    )


def _compare_reducer_head_grads(
    stream_outputs: tuple[torch.Tensor, ...],
    normal_outputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    criterion: nn.Module,
) -> None:
    print("\nReducer-head backward diagnostics:")
    for idx in range(3):
        stream_head = stream_outputs[idx]
        normal_head = normal_outputs[idx]

        stream_head.retain_grad()
        normal_head.retain_grad()

        loss_stream = criterion(stream_head.mean(), target)
        loss_normal = criterion(normal_head.mean(), target)

        stream_grad = torch.autograd.grad(loss_stream, stream_head, retain_graph=True)[0]
        normal_grad = torch.autograd.grad(loss_normal, normal_head, retain_graph=True)[0]

        diff = (stream_grad - normal_grad).abs()
        print(
            f"  reducer_head[{idx}]: "
            f"stream|g|={stream_grad.abs().mean().item():.6e}, "
            f"normal|g|={normal_grad.abs().mean().item():.6e}, "
            f"mean|Δg|={diff.mean().item():.6e}, "
            f"max|Δg|={diff.max().item():.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare streaming vs non-streaming forward/backward behavior for WSS reducer heads."
    )
    parser.add_argument("--dtype", default="float64", help="float16, float32, or float64")
    parser.add_argument("--tile-size", type=int, default=1920)
    parser.add_argument("--input-size", type=int, default=2560)
    parser.add_argument(
        "--reducer-only-backward",
        action="store_true",
        help="Only compare reducer-adjacent backward paths and skip full-model gradient diff.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _parse_dtype(args.dtype)

    img = torch.rand((1, 3, args.input_size, args.input_size), device=device, dtype=dtype)
    target = torch.tensor(50.0, device=device, dtype=dtype)
    criterion = torch.nn.MSELoss()

    network = StreamingWSS(
        "resnet18",
        args.tile_size,
        additional_modules=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
        saliency=True,
    ).to(device=device, dtype=dtype)
    network.stream_network.device = device
    network.stream_network.dtype = dtype
    network.stream_network.mean = network.stream_network.mean.to(device=device, dtype=dtype)
    network.stream_network.std = network.stream_network.std.to(device=device, dtype=dtype)
    _freeze_batchnorm(network.stream_network.stream_module)

    # Streaming path
    _zero_grads(network.stream_network.stream_module.parameters())
    stream_outputs = network(img)
    stream_outputs = tuple(out.requires_grad_(True) for out in stream_outputs)

    # Non-streaming path
    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    _freeze_batchnorm(normal_net)
    _zero_grads(normal_net.parameters())
    img_normal = img.detach().clone().requires_grad_(True)
    normal_outputs = normal_net(img_normal)

    _print_head_forward_stats(stream_outputs, normal_outputs)
    _compare_reducer_head_grads(stream_outputs, normal_outputs, target, criterion)

    # streamed reducer head gradient vs post-reduce(stream_map) gradient consistency
    stream_map = stream_outputs[3]
    normal_map = normal_outputs[3]
    stream_map_grad = torch.autograd.grad(
        criterion(normal_net.reducer(stream_map).mean(), target),
        stream_map,
        retain_graph=True,
    )[0]
    normal_map_grad = torch.autograd.grad(
        criterion(normal_net.reducer(normal_map).mean(), target),
        normal_map,
        retain_graph=True,
    )[0]
    _print_reducer_backward_consistency(stream_map_grad, normal_map_grad)

    # Full backward compares (can be skipped with --reducer-only-backward)
    if not args.reducer_only_backward:
        _zero_grads(network.stream_network.stream_module.parameters())
        stream_pred = [torch.sigmoid(out.mean()) for out in stream_outputs]
        sum(criterion(pred, target) for pred in stream_pred).backward()
        output_grads = tuple(out.grad if out.grad is not None else torch.zeros_like(out) for out in stream_outputs)
        network.stream_network.backward(img, output_grads)
        streaming_all_grads = _gather_param_grads(network.stream_network.stream_module, reducer_only=False)

        _zero_grads(normal_net.parameters())
        normal_pred = [torch.sigmoid(out.mean()) for out in normal_outputs]
        sum(criterion(pred, target) for pred in normal_pred).backward()
        normal_all_grads = _gather_param_grads(normal_net, reducer_only=False)
        _print_grad_stats("Full-model parameter gradient comparison", streaming_all_grads, normal_all_grads)

    _zero_grads(network.stream_network.stream_module.parameters())
    stream_reducer_pred = [torch.sigmoid(out.mean()) for out in stream_outputs[:3]]
    sum(criterion(pred, target) for pred in stream_reducer_pred).backward()
    reducer_output_grads = tuple(out.grad if out.grad is not None else torch.zeros_like(out) for out in stream_outputs)
    network.stream_network.backward(img, reducer_output_grads)
    streaming_reducer_grads = _gather_param_grads(network.stream_network.stream_module, reducer_only=True)

    _zero_grads(normal_net.parameters())
    normal_reducer_pred = [torch.sigmoid(out.mean()) for out in normal_outputs[:3]]
    sum(criterion(pred, target) for pred in normal_reducer_pred).backward()
    normal_reducer_grads = _gather_param_grads(normal_net, reducer_only=True)
    _print_grad_stats("Reducer-adjacent parameter gradient comparison", streaming_reducer_grads, normal_reducer_grads)


if __name__ == "__main__":
    main()
