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
    stream_grads: dict[str, torch.Tensor], normal_grads: dict[str, torch.Tensor], topk: int, title: str
) -> tuple[float, str]:
    shared = sorted(set(stream_grads.keys()) & set(normal_grads.keys()))
    if not shared:
        print(f"No overlapping parameter gradients found to compare for '{title}'.")
        return 0.0, ""

    rows = []
    for name in shared:
        stats = _diff_stats(stream_grads[name], normal_grads[name])
        rows.append((name, stats))

    rows.sort(key=lambda x: x[1]["max"], reverse=True)
    k = min(topk, len(rows))
    print(f"\nBackward diagnostics [{title}] (top {k}/{len(rows)} parameters by max abs diff):")
    worst_mean = 0.0
    worst_name = ""
    for name, stats in rows[:k]:
        _print_stats(f"grad[{name}]", stats)
        if stats["mean"] > worst_mean:
            worst_mean = stats["mean"]
            worst_name = f"grad[{name}]"
    return worst_mean, worst_name


def _losses_model_heads(outputs: list[torch.Tensor], reducer: GlobalReducer, criterion: nn.Module) -> list[torch.Tensor]:
    if len(outputs) < 4:
        raise ValueError(f"Expected at least 4 outputs from WSS, got {len(outputs)}")
    y1_r, y2_r, y3_r, y = outputs[:4]
    y_reduced = reducer(y)
    return [
        criterion(y1_r, torch.ones_like(y1_r)),
        criterion(y2_r, torch.ones_like(y2_r)),
        criterion(y3_r, torch.ones_like(y3_r)),
        criterion(y_reduced, torch.ones_like(y_reduced)),
    ]


def _losses_post_reduce_maps(outputs: list[torch.Tensor], reducer: GlobalReducer, criterion: nn.Module) -> list[torch.Tensor]:
    if len(outputs) < 7:
        raise ValueError("Post-reduce map diagnostics require 7 outputs: reduced heads + y + raw maps.")
    y, y1, y2, y3 = outputs[3], outputs[4], outputs[5], outputs[6]
    return [
        criterion(reducer(y1), torch.ones_like(outputs[0])),
        criterion(reducer(y2), torch.ones_like(outputs[1])),
        criterion(reducer(y3), torch.ones_like(outputs[2])),
        criterion(reducer(y), torch.ones_like(outputs[0])),
    ]




def _reduced_outputs_global(outputs: list[torch.Tensor], reducer: GlobalReducer) -> list[torch.Tensor]:
    if len(outputs) < 4:
        raise ValueError("Need at least 4 outputs for global-reduce view")
    return [outputs[0], outputs[1], outputs[2], reducer(outputs[3])]


def _reduced_outputs_post(outputs: list[torch.Tensor], reducer: GlobalReducer) -> list[torch.Tensor]:
    if len(outputs) < 7:
        raise ValueError("Need 7 outputs for post-reduce view")
    return [reducer(outputs[4]), reducer(outputs[5]), reducer(outputs[6]), reducer(outputs[3])]


def _compare_output_sets(name_a: str, outs_a: list[torch.Tensor], name_b: str, outs_b: list[torch.Tensor]) -> tuple[float, str]:
    if len(outs_a) != len(outs_b):
        raise ValueError(f"Output set size mismatch: {name_a}={len(outs_a)} vs {name_b}={len(outs_b)}")

    print(f"\nForward pairwise [{name_a}] vs [{name_b}]:")
    worst_mean = 0.0
    worst_name = ""
    for idx, (a, b) in enumerate(zip(outs_a, outs_b)):
        stats = _diff_stats(a, b)
        _print_stats(f"pair[{idx}]", stats)
        if stats["mean"] > worst_mean:
            worst_mean = stats["mean"]
            worst_name = f"forward[{name_a} vs {name_b}][{idx}]"
    return worst_mean, worst_name


def _loss_grads_for_outputs(
    outputs: list[torch.Tensor],
    losses_builder,
    reducer: GlobalReducer,
    criterion: nn.Module,
    required_grad_indices: set[int],
) -> list[torch.Tensor]:
    prepared: list[torch.Tensor] = []
    for out in outputs:
        out.requires_grad_(True)
        out.retain_grad()
        prepared.append(out)

    losses = losses_builder(prepared, reducer, criterion)
    sum(losses).backward()

    grads: list[torch.Tensor] = []
    for idx, out in enumerate(prepared):
        grad = out.grad
        if grad is None:
            if idx in required_grad_indices:
                raise RuntimeError(f"Missing required output gradient at index {idx}.")
            grad = torch.zeros_like(out)
        grads.append(grad)
    return grads


def _compare_gradient_sets(
    grad_set_a: dict[str, torch.Tensor],
    grad_set_b: dict[str, torch.Tensor],
    title: str,
) -> tuple[float, str]:
    shared = sorted(set(grad_set_a) & set(grad_set_b))
    if not shared:
        return 0.0, ""

    worst_mean = 0.0
    worst_max = 0.0
    worst_name = ""
    for name in shared:
        stats = _diff_stats(grad_set_a[name], grad_set_b[name])
        if stats["mean"] > worst_mean:
            worst_mean = stats["mean"]
            worst_max = stats["max"]
            worst_name = f"{title}[{name}]"

    print(f"\nBackward pairwise [{title}]:")
    _print_stats("worst", {"mean": worst_mean, "max": worst_max})
    return worst_mean, worst_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare streaming vs non-streaming WSS (with reducer diagnostics).")
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

    network.stream_network.stream_module.eval()
    _freeze_batchnorm(network.stream_network.stream_module)

    # 1) Streaming operations first.
    with torch.no_grad():
        streaming_outputs = _to_sequence(network(image))

    stream_grads_heads: dict[str, torch.Tensor] | None = None
    stream_grads_post: dict[str, torch.Tensor] | None = None

    if args.debug_backward:
        criterion = nn.BCELoss().to(device=device)

        # Scenario A: use model-reduced heads [0:3]
        network.stream_network.enable()
        _zero_grads(network.stream_network.stream_module)
        stream_input_a = image.detach().clone()
        stream_outputs_a = _to_sequence(network(stream_input_a))
        reducer_a = GlobalReducer().to(device=device, dtype=dtype)
        grads_a = _loss_grads_for_outputs(stream_outputs_a, _losses_model_heads, reducer_a, criterion, {0, 1, 2, 3})
        network.stream_network.backward(stream_input_a, tuple(grads_a))
        stream_grads_heads = _collect_grads(network.stream_network.stream_module)

        # Scenario B: use raw maps [4:6], reduce after streaming
        network.stream_network.enable()
        _zero_grads(network.stream_network.stream_module)
        stream_input_b = image.detach().clone()
        stream_outputs_b = _to_sequence(network(stream_input_b))
        reducer_b = GlobalReducer().to(device=device, dtype=dtype)
        grads_b = _loss_grads_for_outputs(stream_outputs_b, _losses_post_reduce_maps, reducer_b, criterion, {3, 4, 5, 6})
        network.stream_network.backward(stream_input_b, tuple(grads_b))
        stream_grads_post = _collect_grads(network.stream_network.stream_module)

    # 2) Then non-streaming operations.
    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    normal_net.eval()
    _freeze_batchnorm(normal_net)

    with torch.no_grad():
        normal_outputs = _to_sequence(normal_net(image))

    normal_grads_heads: dict[str, torch.Tensor] | None = None
    normal_grads_post: dict[str, torch.Tensor] | None = None

    if args.debug_backward:
        criterion = nn.BCELoss().to(device=device)

        # Scenario A
        _zero_grads(normal_net)
        normal_input_a = image.detach().clone().requires_grad_(True)
        normal_outputs_a = _to_sequence(normal_net(normal_input_a))
        reducer_na = GlobalReducer().to(device=device, dtype=dtype)
        losses_na = _losses_model_heads(normal_outputs_a, reducer_na, criterion)
        sum(losses_na).backward()
        normal_grads_heads = _collect_grads(normal_net)

        # Scenario B
        _zero_grads(normal_net)
        normal_input_b = image.detach().clone().requires_grad_(True)
        normal_outputs_b = _to_sequence(normal_net(normal_input_b))
        reducer_nb = GlobalReducer().to(device=device, dtype=dtype)
        losses_nb = _losses_post_reduce_maps(normal_outputs_b, reducer_nb, criterion)
        sum(losses_nb).backward()
        normal_grads_post = _collect_grads(normal_net)

    # 3) Compare / report
    if len(streaming_outputs) != len(normal_outputs):
        raise ValueError(f"Output count mismatch: streaming={len(streaming_outputs)}, non-streaming={len(normal_outputs)}")

    print(f"Compared {len(streaming_outputs)} outputs ({args.encoder}, {dtype}, input={args.input_size}, tile={args.tile_size})")
    worst_mean = 0.0
    worst_name = ""

    reducer_view = GlobalReducer().to(device=device, dtype=dtype)
    forward_sets = {
        "streaming global reduce": _reduced_outputs_global(streaming_outputs, reducer_view),
        "streaming post reduce": _reduced_outputs_post(streaming_outputs, reducer_view),
        "normal global reduce": _reduced_outputs_global(normal_outputs, reducer_view),
        "normal post reduce": _reduced_outputs_post(normal_outputs, reducer_view),
    }

    names = list(forward_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m, n = _compare_output_sets(names[i], forward_sets[names[i]], names[j], forward_sets[names[j]])
            if m > worst_mean:
                worst_mean, worst_name = m, n

    if args.debug_backward and all(x is not None for x in [stream_grads_heads, normal_grads_heads, stream_grads_post, normal_grads_post]):
        backward_sets = {
            "streaming global reduce": stream_grads_heads,
            "streaming post reduce": stream_grads_post,
            "normal global reduce": normal_grads_heads,
            "normal post reduce": normal_grads_post,
        }

        for key, (a, b) in {
            "streaming global reduce": (stream_grads_heads, normal_grads_heads),
            "streaming post reduce": (stream_grads_post, normal_grads_post),
        }.items():
            mm, grad_name = _compare_grads(a, b, args.backward_topk, title=key)
            if mm > worst_mean:
                worst_mean, worst_name = mm, grad_name

        bnames = list(backward_sets.keys())
        for i in range(len(bnames)):
            for j in range(i + 1, len(bnames)):
                sm, sn = _compare_gradient_sets(
                    backward_sets[bnames[i]],
                    backward_sets[bnames[j]],
                    f"{bnames[i]} vs {bnames[j]}",
                )
                if sm > worst_mean:
                    worst_mean, worst_name = sm, sn

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
