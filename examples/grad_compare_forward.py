from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from lightstream.models.segment.reducer import GlobalReducer
from lightstream.models.segment.streamingwss import StreamingWSS


def _parse_dtype(value: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}
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
    d = (a - b).abs()
    return {"mean": d.mean().item(), "max": d.max().item()}


def _print_stats(prefix: str, stats: dict[str, float]) -> None:
    print(f"{prefix}: mean abs diff={stats['mean']:.6e}, max abs diff={stats['max']:.6e}")


def _zero_grads(module: nn.Module) -> None:
    for p in module.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _collect_grads(module: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for n, p in module.named_parameters():
        if p.grad is not None:
            out[n] = p.grad.detach().clone()
    return out


def _compare_grads(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], topk: int, title: str) -> tuple[float, str]:
    shared = sorted(set(a) & set(b))
    rows = [(n, _diff_stats(a[n], b[n])) for n in shared]
    rows.sort(key=lambda x: x[1]["max"], reverse=True)
    k = min(topk, len(rows))
    print(f"\nBackward diagnostics [{title}] (top {k}/{len(rows)}):")
    worst_mean, worst_name = 0.0, ""
    for n, st in rows[:k]:
        _print_stats(f"grad[{n}]", st)
        if st["mean"] > worst_mean:
            worst_mean, worst_name = st["mean"], f"grad[{n}]"
    return worst_mean, worst_name


def _losses_reduced(outputs: list[torch.Tensor], reducer: GlobalReducer, criterion: nn.Module) -> list[torch.Tensor]:
    y1r, y2r, y3r, y = outputs
    return [
        criterion(y1r, torch.ones_like(y1r)),
        criterion(y2r, torch.ones_like(y2r)),
        criterion(y3r, torch.ones_like(y3r)),
        criterion(reducer(y), torch.ones_like(y1r)),
    ]


def _losses_raw(outputs: list[torch.Tensor], reducer: GlobalReducer, criterion: nn.Module) -> list[torch.Tensor]:
    y1, y2, y3, y = outputs
    return [
        criterion(reducer(y1), torch.ones((y1.shape[0], y1.shape[1]), device=y1.device, dtype=y1.dtype)),
        criterion(reducer(y2), torch.ones((y2.shape[0], y2.shape[1]), device=y2.device, dtype=y2.dtype)),
        criterion(reducer(y3), torch.ones((y3.shape[0], y3.shape[1]), device=y3.device, dtype=y3.dtype)),
        criterion(reducer(y), torch.ones((y.shape[0], y.shape[1]), device=y.device, dtype=y.dtype)),
    ]


def _output_grads(outputs: list[torch.Tensor], losses_builder, reducer: GlobalReducer, criterion: nn.Module) -> list[torch.Tensor]:
    for o in outputs:
        o.requires_grad_(True)
        o.retain_grad()
    losses = losses_builder(outputs, reducer, criterion)
    sum(losses).backward()
    grads = []
    for o in outputs:
        grads.append(torch.zeros_like(o) if o.grad is None else o.grad)
    return grads


def _configure_stream_model(net: StreamingWSS, device: torch.device, dtype: torch.dtype) -> None:
    net.stream_network.device = device
    net.stream_network.dtype = dtype
    net.stream_network.mean = net.stream_network.mean.to(device=device, dtype=dtype)
    net.stream_network.std = net.stream_network.std.to(device=device, dtype=dtype)
    net.stream_network.stream_module.eval()
    _freeze_batchnorm(net.stream_network.stream_module)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="resnet18")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--tile-size", type=int, default=1920)
    p.add_argument("--input-size", type=int, default=2560)
    p.add_argument("--debug-backward", action="store_true")
    p.add_argument("--backward-topk", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _parse_dtype(args.dtype)
    image = torch.rand((1, 3, args.input_size, args.input_size), device=device, dtype=dtype)

    cache_root = Path(__file__).parent
    nets = {
        "streaming global reduce": StreamingWSS(
            args.encoder,
            args.tile_size,
            mean=[0, 0, 0],
            std=[1, 1, 1],
            normalize_on_gpu=False,
            saliency=False,
            model_kind="reduced",
            tile_cache_path=cache_root / f"{args.encoder}_reduced_tile_cache_1_3_{args.tile_size}_{args.tile_size}",
        ).to(device=device, dtype=dtype),
        "streaming post reduce": StreamingWSS(
            args.encoder,
            args.tile_size,
            mean=[0, 0, 0],
            std=[1, 1, 1],
            normalize_on_gpu=False,
            saliency=False,
            model_kind="raw",
            tile_cache_path=cache_root / f"{args.encoder}_raw_tile_cache_1_3_{args.tile_size}_{args.tile_size}",
        ).to(device=device, dtype=dtype),
    }
    for net in nets.values():
        _configure_stream_model(net, device, dtype)

    with torch.no_grad():
        s_reduced_out = _to_sequence(nets["streaming global reduce"](image))
        s_raw_out = _to_sequence(nets["streaming post reduce"](image))

    # switch to normal versions (separate modules, separate stats)
    for net in nets.values():
        net.stream_network.disable()
    n_reduced = nets["streaming global reduce"].stream_network.stream_module
    n_raw = nets["streaming post reduce"].stream_network.stream_module
    _freeze_batchnorm(n_reduced)
    _freeze_batchnorm(n_raw)
    n_reduced.eval(); n_raw.eval()

    with torch.no_grad():
        n_reduced_out = _to_sequence(n_reduced(image))
        n_raw_out = _to_sequence(n_raw(image))

    reducer = GlobalReducer().to(device=device, dtype=dtype)
    forward_sets = {
        "streaming global reduce": [s_reduced_out[0], s_reduced_out[1], s_reduced_out[2], reducer(s_reduced_out[3])],
        "streaming post reduce": [reducer(s_raw_out[0]), reducer(s_raw_out[1]), reducer(s_raw_out[2]), reducer(s_raw_out[3])],
        "normal global reduce": [n_reduced_out[0], n_reduced_out[1], n_reduced_out[2], reducer(n_reduced_out[3])],
        "normal post reduce": [reducer(n_raw_out[0]), reducer(n_raw_out[1]), reducer(n_raw_out[2]), reducer(n_raw_out[3])],
    }

    names = list(forward_sets.keys())
    worst_mean, worst_name = 0.0, ""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"\nForward pairwise [{names[i]}] vs [{names[j]}]:")
            for idx, (a, b) in enumerate(zip(forward_sets[names[i]], forward_sets[names[j]])):
                st = _diff_stats(a, b)
                _print_stats(f"pair[{idx}]", st)
                if st["mean"] > worst_mean:
                    worst_mean, worst_name = st["mean"], f"forward[{names[i]} vs {names[j]}][{idx}]"

    if args.debug_backward:
        criterion = nn.BCELoss().to(device=device)

        # streaming reduced
        net_sr = nets["streaming global reduce"]
        net_sr.stream_network.enable(); _zero_grads(net_sr.stream_network.stream_module)
        sr_out = _to_sequence(net_sr(image.detach().clone()))
        sr_grads = _output_grads(sr_out, _losses_reduced, GlobalReducer().to(device=device, dtype=dtype), criterion)
        net_sr.stream_network.backward(image.detach().clone(), tuple(sr_grads))
        g_sr = _collect_grads(net_sr.stream_network.stream_module)

        # streaming raw
        net_sp = nets["streaming post reduce"]
        net_sp.stream_network.enable(); _zero_grads(net_sp.stream_network.stream_module)
        sp_out = _to_sequence(net_sp(image.detach().clone()))
        sp_grads = _output_grads(sp_out, _losses_raw, GlobalReducer().to(device=device, dtype=dtype), criterion)
        net_sp.stream_network.backward(image.detach().clone(), tuple(sp_grads))
        g_sp = _collect_grads(net_sp.stream_network.stream_module)

        # normal reduced/raw (ensure wrappers are disabled after streaming debug passes)
        net_sr.stream_network.disable()
        net_sp.stream_network.disable()
        n_reduced = net_sr.stream_network.stream_module
        n_raw = net_sp.stream_network.stream_module
        n_reduced.eval(); n_raw.eval()
        _freeze_batchnorm(n_reduced)
        _freeze_batchnorm(n_raw)

        _zero_grads(n_reduced)
        nr_out = _to_sequence(n_reduced(image.detach().clone().requires_grad_(True)))
        sum(_losses_reduced(nr_out, GlobalReducer().to(device=device, dtype=dtype), criterion)).backward()
        g_nr = _collect_grads(n_reduced)

        _zero_grads(n_raw)
        np_out = _to_sequence(n_raw(image.detach().clone().requires_grad_(True)))
        sum(_losses_raw(np_out, GlobalReducer().to(device=device, dtype=dtype), criterion)).backward()
        g_np = _collect_grads(n_raw)

        backward_sets = {
            "streaming global reduce": g_sr,
            "streaming post reduce": g_sp,
            "normal global reduce": g_nr,
            "normal post reduce": g_np,
        }
        bnames = list(backward_sets.keys())
        for i in range(len(bnames)):
            for j in range(i + 1, len(bnames)):
                m, n = _compare_grads(backward_sets[bnames[i]], backward_sets[bnames[j]], args.backward_topk, f"{bnames[i]} vs {bnames[j]}")
                if m > worst_mean:
                    worst_mean, worst_name = m, n

    print(f"\nWorst mean abs diff: {worst_name} = {worst_mean:.6e}")


if __name__ == "__main__":
    main()
