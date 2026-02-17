from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch.nn import Sequential
from torchvision.models import resnet18, resnet34, resnet50

from lightstream.models.segment.model import WSS
from lightstream.modules.loss_reducer import GlobalWSLossReducer, StreamingGlobalWSLossReducer
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


def _compare_grads(stream_grads: dict[str, torch.Tensor], normal_grads: dict[str, torch.Tensor]) -> None:
    shared = sorted(set(stream_grads.keys()) & set(normal_grads.keys()))
    if not shared:
        print("No overlapping gradients found to compare.")
        return

    print(f"Comparing {len(shared)} parameter gradients:")
    for name in shared:
        diff = (stream_grads[name] - normal_grads[name]).abs()
        print(f"{name}: mean abs diff={diff.mean().item():.6e}, max abs diff={diff.max().item():.6e}")


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


def _stream_reduce_score(
    logits: torch.Tensor,
    reducer: StreamingGlobalWSLossReducer,
    tile_h: int,
    tile_w: int,
) -> torch.Tensor:
    """Run reducer tile-wise on a full map to emulate streaming supervision accumulation."""
    _, _, height, width = logits.shape
    reducer.reset(spatial_shape=(height, width))

    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            y2 = min(y + tile_h, height)
            x2 = min(x + tile_w, width)
            tile = logits[:, :, y:y2, x:x2]
            reducer.update(tile, tile_origin=(y, x))

    return reducer.pooled_score()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare streaming vs non-streaming gradients for WSS with global-reducer side supervision."
    )
    parser.add_argument("--dtype", default="float64", help="float16, float32, or float64")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--input-size", type=int, default=1536)
    parser.add_argument("--reduce-tile-size", type=int, default=512)
    parser.add_argument("--r", type=float, default=4.0)
    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _parse_dtype(args.dtype)

    img = torch.rand((1, 3, args.input_size, args.input_size), device=device, dtype=dtype)
    slide_label = torch.tensor([[1.0]], device=device, dtype=dtype)

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

    # --- Streaming path ---
    _zero_grads(network.stream_network.stream_module.parameters())
    stream_outputs = network(img)
    y1_s, y2_s, y3_s, y_s = stream_outputs

    for out in stream_outputs:
        out.retain_grad()

    reducer_global = GlobalWSLossReducer(r=args.r)
    reducer1_stream = StreamingGlobalWSLossReducer(r=args.r)
    reducer2_stream = StreamingGlobalWSLossReducer(r=args.r)
    reducer3_stream = StreamingGlobalWSLossReducer(r=args.r)

    # side outputs reduced to slide-level scalars using streaming reducers
    s1 = _stream_reduce_score(y1_s, reducer1_stream, args.reduce_tile_size, args.reduce_tile_size)
    s2 = _stream_reduce_score(y2_s, reducer2_stream, args.reduce_tile_size, args.reduce_tile_size)
    s3 = _stream_reduce_score(y3_s, reducer3_stream, args.reduce_tile_size, args.reduce_tile_size)

    # final map supervision stays spatial (example uses same reducer for simplicity)
    y_score_stream = reducer_global.aggregate(y_s)

    stream_loss = (
        nn.functional.binary_cross_entropy(s1, slide_label)
        + nn.functional.binary_cross_entropy(s2, slide_label)
        + nn.functional.binary_cross_entropy(s3, slide_label)
        + nn.functional.binary_cross_entropy(y_score_stream, slide_label)
    )
    stream_loss.backward()

    output_grads = tuple(out.grad if out.grad is not None else torch.zeros_like(out) for out in stream_outputs)
    network.stream_network.backward(img, output_grads)
    streaming_param_grads = _gather_param_grads(network.stream_network.stream_module)

    # --- Non-streaming reference path ---
    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    _freeze_batchnorm(normal_net)
    _zero_grads(normal_net.parameters())

    img_normal = img.detach().clone().requires_grad_(True)
    y1_n, y2_n, y3_n, y_n = normal_net(img_normal)

    for stream_out, normal_out in zip(stream_outputs, (y1_n, y2_n, y3_n, y_n)):
        diff = (stream_out - normal_out).abs()
        print(f"Forward output sum/max diff: {diff.sum().item():.6e}, {diff.max().item():.6e}")

    s1_n = reducer_global.aggregate(y1_n)
    s2_n = reducer_global.aggregate(y2_n)
    s3_n = reducer_global.aggregate(y3_n)
    y_score_normal = reducer_global.aggregate(y_n)

    normal_loss = (
        nn.functional.binary_cross_entropy(s1_n, slide_label)
        + nn.functional.binary_cross_entropy(s2_n, slide_label)
        + nn.functional.binary_cross_entropy(s3_n, slide_label)
        + nn.functional.binary_cross_entropy(y_score_normal, slide_label)
    )
    normal_loss.backward()

    normal_param_grads = _gather_param_grads(normal_net)

    if img_normal.grad is not None:
        input_grad_diff = img_normal.grad.detach().cpu() - network.stream_network.saliency_map[0]
        print(f"Input gradient max diff: {input_grad_diff.abs().max().item():.6e}")

    _compare_grads(streaming_param_grads, normal_param_grads)


if __name__ == "__main__":
    main()
