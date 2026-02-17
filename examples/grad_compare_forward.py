from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from lightstream.models.segment.streamingwss import StreamingWSS


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


def _to_sequence(outputs):
    if isinstance(outputs, (tuple, list)):
        return list(outputs)
    return [outputs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Forward-only comparison for streaming vs non-streaming WSS outputs. "
            "(No backward comparison yet for GlobalReducer.)"
        )
    )
    parser.add_argument("--encoder", default="resnet18", help="resnet18, resnet34, or resnet50")
    parser.add_argument("--dtype", default="float64", help="float16, float32, or float64")
    parser.add_argument("--tile-size", type=int, default=1920)
    parser.add_argument("--input-size", type=int, default=2560)
    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _parse_dtype(args.dtype)

    image = torch.rand((1, 3, args.input_size, args.input_size), device=device, dtype=dtype)

    network = StreamingWSS(
        encoder=args.encoder,
        tile_size=args.tile_size,
        additional_modules=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
        saliency=False,
    ).to(device=device, dtype=dtype)

    network.stream_network.device = device
    network.stream_network.dtype = dtype
    network.stream_network.mean = network.stream_network.mean.to(device=device, dtype=dtype)
    network.stream_network.std = network.stream_network.std.to(device=device, dtype=dtype)

    _freeze_batchnorm(network.stream_network.stream_module)

    with torch.no_grad():
        streaming_outputs = _to_sequence(network(image))

        network.stream_network.disable()
        normal_net = network.stream_network.stream_module
        _freeze_batchnorm(normal_net)
        normal_outputs = _to_sequence(normal_net(image))

    if len(streaming_outputs) != len(normal_outputs):
        raise ValueError(
            f"Output count mismatch: streaming={len(streaming_outputs)}, non-streaming={len(normal_outputs)}"
        )

    print(f"Compared {len(streaming_outputs)} outputs ({args.encoder}, {dtype}, input={args.input_size}, tile={args.tile_size})")
    for idx, (stream_out, normal_out) in enumerate(zip(streaming_outputs, normal_outputs)):
        diff = (stream_out - normal_out).abs()
        print(
            f"output[{idx}]: "
            f"mean abs diff={diff.mean().item():.6e}, "
            f"max abs diff={diff.max().item():.6e}, "
            f"sum abs diff={diff.sum().item():.6e}"
        )


if __name__ == "__main__":
    main()
