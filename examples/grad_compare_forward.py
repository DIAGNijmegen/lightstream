from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from lightstream.models.segment.reducer import GlobalReducer
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


def _diff_stats(a: torch.Tensor, b: torch.Tensor):
    diff = (a - b).abs()
    return {
        "mean": diff.mean().item(),
        "max": diff.max().item(),
    }


def _print_stats(prefix: str, stats: dict[str, float]) -> None:
    print(f"{prefix}: mean abs diff={stats['mean']:.6e}, max abs diff={stats['max']:.6e}")


def _zero_grads(module: nn.Module) -> None:
    for param in module.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def _collect_grads(module: nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        if param.grad is not None:
            out[name] = param.grad.detach().clone()
    return out


def _compare_grads(
    stream_grads: dict[str, torch.Tensor], normal_grads: dict[str, torch.Tensor], topk: int
) -> tuple[float, str]:
    shared = sorted(set(stream_grads.keys()) & set(normal_grads.keys()))
    if not shared:
        print("No overlapping parameter gradients found to compare.")
        return 0.0, ""

    rows = []
    for name in shared:
        stats = _diff_stats(stream_grads[name], normal_grads[name])
        rows.append((name, stats))

    rows.sort(key=lambda x: x[1]["max"], reverse=True)
    k = min(topk, len(rows))
    print(f"\nBackward diagnostics (top {k}/{len(rows)} parameters by max abs diff):")
    worst_mean = 0.0
    worst_name = ""
    for name, stats in rows[:k]:
        _print_stats(f"grad[{name}]", stats)
        if stats["mean"] > worst_mean:
            worst_mean = stats["mean"]
            worst_name = f"grad[{name}]"
    return worst_mean, worst_name


def _build_losses(outputs: list[torch.Tensor], reducer: GlobalReducer, criterion: nn.Module) -> list[torch.Tensor]:
    if len(outputs) != 4:
        raise ValueError(f"Expected 4 outputs from WSS, got {len(outputs)}")

    y1, y2, y3, y = outputs
    y_reduced = reducer(y)

    return [
        criterion(y1, torch.ones_like(y1)),
        criterion(y2, torch.ones_like(y2)),
        criterion(y3, torch.ones_like(y3)),
        criterion(y_reduced, torch.ones_like(y_reduced)),
    ]


def _loss_grads_for_outputs(
    outputs: list[torch.Tensor], reducer: GlobalReducer, criterion: nn.Module
) -> list[torch.Tensor]:
    detached = [out.detach().clone().requires_grad_(True) for out in outputs]
    losses = _build_losses(detached, reducer, criterion)
    sum(losses).backward()
    grads = [out.grad for out in detached]
    if any(grad is None for grad in grads):
        raise RuntimeError("Missing output gradient(s) while building loss gradients.")
    return grads


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare streaming vs non-streaming WSS on all 4 outputs and run backward diagnostics. "
            "Backward uses BCE target=1 on y1/y2/y3 and on GlobalReducer(y)."
        )
    )
    parser.add_argument("--encoder", default="resnet18", help="resnet18, resnet34, or resnet50")
    parser.add_argument("--dtype", default="float64", help="float16, float32, or float64")
    parser.add_argument("--tile-size", type=int, default=1920)
    parser.add_argument("--input-size", type=int, default=2560)
    parser.add_argument("--warn-mean-threshold", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--debug-backward", action="store_true")
    parser.add_argument("--backward-topk", type=int, default=20)
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

    # 1) Do all streaming operations first.
    with torch.no_grad():
        streaming_outputs = _to_sequence(network(image))

    stream_grads: dict[str, torch.Tensor] | None = None
    if args.debug_backward:
        criterion = nn.BCELoss().to(device=device)

        network.stream_network.enable()
        _freeze_batchnorm(network.stream_network.stream_module)
        _zero_grads(network.stream_network.stream_module)

        stream_input = image.detach().clone()
        stream_outputs_train = _to_sequence(network(stream_input))
        stream_output_grads = _loss_grads_for_outputs(
            stream_outputs_train,
            GlobalReducer().to(device=device, dtype=dtype),
            criterion,
        )
        network.stream_network.backward(stream_input, stream_output_grads)
        stream_grads = _collect_grads(network.stream_network.stream_module)

    # 2) Then switch to normal and do all normal operations.
    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    _freeze_batchnorm(normal_net)

    with torch.no_grad():
        normal_outputs = _to_sequence(normal_net(image))

    normal_grads: dict[str, torch.Tensor] | None = None
    if args.debug_backward:
        criterion = nn.BCELoss().to(device=device)

        _zero_grads(normal_net)
        normal_input = image.detach().clone().requires_grad_(True)
        normal_outputs_train = _to_sequence(normal_net(normal_input))
        normal_output_grads = _loss_grads_for_outputs(
            normal_outputs_train,
            GlobalReducer().to(device=device, dtype=dtype),
            criterion,
        )
        torch.autograd.backward(normal_outputs_train, grad_tensors=normal_output_grads)
        normal_grads = _collect_grads(normal_net)

    # 3) Finally compare and print differences.
    if len(streaming_outputs) != len(normal_outputs):
        raise ValueError(
            f"Output count mismatch: streaming={len(streaming_outputs)}, non-streaming={len(normal_outputs)}"
        )

    print(f"Compared {len(streaming_outputs)} outputs ({args.encoder}, {dtype}, input={args.input_size}, tile={args.tile_size})")
    worst_mean = 0.0
    worst_name = ""
    for idx, (stream_out, normal_out) in enumerate(zip(streaming_outputs, normal_outputs)):
        stats = _diff_stats(stream_out, normal_out)
        _print_stats(f"output[{idx}] stream-vs-normal", stats)
        if stats["mean"] > worst_mean:
            worst_mean = stats["mean"]
            worst_name = f"output[{idx}] stream-vs-normal"

    if args.debug_backward and stream_grads is not None and normal_grads is not None:
        grad_worst_mean, grad_worst_name = _compare_grads(stream_grads, normal_grads, args.backward_topk)
        if grad_worst_mean > worst_mean:
            worst_mean = grad_worst_mean
            worst_name = grad_worst_name

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
