from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from lightstream.models.segment.streamingwss import StreamingWSS
from lightstream.models.segment.reducer import GlobalReducer


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


def _diff_stats(a: torch.Tensor, b: torch.Tensor, eps: float):
    diff = (a - b).abs()
    denom = b.abs().mean().clamp_min(eps)
    return {
        "mean": diff.mean().item(),
        "max": diff.max().item(),
        "sum": diff.sum().item(),
        "rel_mean": (diff.mean() / denom).item(),
    }


def _print_stats(prefix: str, stats: dict[str, float]) -> None:
    print(
        f"{prefix}: "
        f"mean abs diff={stats['mean']:.6e}, "
        f"max abs diff={stats['max']:.6e}, "
        f"sum abs diff={stats['sum']:.6e}, "
        f"rel mean diff={stats['rel_mean']:.6e}"
    )


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
    parser.add_argument("--eps", type=float, default=1e-12, help="epsilon for relative-difference denominator")
    parser.add_argument(
        "--warn-mean-threshold",
        type=float,
        default=0.0,
        help="warn if any output mean absolute difference is above this value",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with code 1 when warn threshold is exceeded",
    )
    parser.add_argument(
        "--skip-reducer-diagnostics",
        action="store_true",
        help="skip reducer-vs-post-reduce diagnostic section",
    )
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
    worst_mean = 0.0
    worst_name = ""
    for idx, (stream_out, normal_out) in enumerate(zip(streaming_outputs, normal_outputs)):
        stats = _diff_stats(stream_out, normal_out, args.eps)
        _print_stats(f"output[{idx}]", stats)
        if stats["mean"] > worst_mean:
            worst_mean = stats["mean"]
            worst_name = f"output[{idx}]"

    # Extra reducer diagnostics:
    # Compare streaming reducer heads [0:3] against reducing streamed feature maps [4:7]
    if len(streaming_outputs) >= 7 and not args.skip_reducer_diagnostics:
        post_reducer = GlobalReducer().to(device=device)
        post_reduce_stream = [
            post_reducer(streaming_outputs[4]),
            post_reducer(streaming_outputs[5]),
            post_reducer(streaming_outputs[6]),
        ]
        post_reduce_normal = [
            post_reducer(normal_outputs[4]),
            post_reducer(normal_outputs[5]),
            post_reducer(normal_outputs[6]),
        ]

        print("\nReducer diagnostics (head reducer vs post-reduce on feature map):")
        for idx in range(3):
            head_stream = streaming_outputs[idx]
            head_normal = normal_outputs[idx]
            stream_post = post_reduce_stream[idx]
            normal_post = post_reduce_normal[idx]

            stats_head_vs_post_stream = _diff_stats(head_stream, stream_post, args.eps)
            stats_head_vs_post_normal = _diff_stats(head_normal, normal_post, args.eps)
            stats_post_stream_vs_normal = _diff_stats(stream_post, normal_post, args.eps)

            _print_stats(f"head[{idx}] stream-vs-post(stream_map)", stats_head_vs_post_stream)
            _print_stats(f"head[{idx}] normal-vs-post(normal_map)", stats_head_vs_post_normal)
            _print_stats(f"head[{idx}] post(stream_map)-vs-post(normal_map)", stats_post_stream_vs_normal)

            if stats_head_vs_post_stream["mean"] > worst_mean:
                worst_mean = stats_head_vs_post_stream["mean"]
                worst_name = f"head[{idx}] stream-vs-post(stream_map)"

    print(f"\nWorst mean abs diff: {worst_name} = {worst_mean:.6e}")
    if args.warn_mean_threshold > 0 and worst_mean > args.warn_mean_threshold:
        message = (
            f"Mean diff threshold exceeded: worst={worst_mean:.6e} > "
            f"warn_mean_threshold={args.warn_mean_threshold:.6e} ({worst_name})"
        )
        if args.strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


if __name__ == "__main__":
    main()
