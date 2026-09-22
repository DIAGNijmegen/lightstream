"""Compare NHWC, NCHW, and streamed NAT feature-map gradients.

Examples
--------
Random, identically initialized weights::

    python examples/grad_compare_natformer.py --variant nat_mini

An official ImageNet checkpoint already downloaded locally::

    python examples/grad_compare_natformer.py --variant nat_mini --checkpoint nat_mini.pth

Download the official checkpoint (the equivalent of ``pretrained=True``)::

    python examples/grad_compare_natformer.py --variant nat_mini --pretrained

To deliberately regenerate statistics, add ``--fresh-tile-statistics``.  Without
that flag ``--tile-cache PATH`` loads a compatible existing cache (and creates
and saves it when it does not yet exist).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from lightstream.models.nat.nat import NAT, model_urls
from lightstream.models.nat.nchw import (
    NCHWNatBase,
    NCHWNatMini,
    NCHWNatSmall,
    convert_nhwc_nat_state_dict,
)
from lightstream.models.nat.streaming import StreamingNAT


VARIANTS = {
    "nat_mini": dict(depths=[3, 4, 6, 5], num_heads=[2, 4, 8, 16], embed_dim=64,
                     mlp_ratio=3, kernel_size=7, layer_scale=None,
                     nchw=NCHWNatMini, checkpoint="nat_mini_1k"),
    "nat_small": dict(depths=[3, 4, 18, 5], num_heads=[3, 6, 12, 24], embed_dim=96,
                      mlp_ratio=2, kernel_size=7, layer_scale=1e-5,
                      nchw=NCHWNatSmall, checkpoint="nat_small_1k"),
    "nat_base": dict(depths=[3, 4, 18, 5], num_heads=[4, 8, 16, 32], embed_dim=128,
                     mlp_ratio=2, kernel_size=7, layer_scale=1e-5,
                     nchw=NCHWNatBase, checkpoint="nat_base_1k"),
}


def _checkpoint(selection: str | None, pretrained: bool, variant: str) -> Mapping[str, torch.Tensor] | None:
    if selection is None and not pretrained:
        return None
    source = model_urls[VARIANTS[variant]["checkpoint"]] if pretrained else selection
    assert source is not None
    if source.startswith(("http://", "https://")):
        state = torch.hub.load_state_dict_from_url(source, map_location="cpu")
    else:
        state = torch.load(source, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    return state


def _reference(variant: str) -> NAT:
    config = {key: value for key, value in VARIANTS[variant].items()
              if key not in {"nchw", "checkpoint"}}
    return NAT(**config, num_classes=1000, drop_rate=0.0, attn_drop_rate=0.0,
               drop_path_rate=0.0)


def _strict_load(model: nn.Module, state: Mapping[str, torch.Tensor], label: str) -> None:
    incompatible = model.load_state_dict(state, strict=True)
    print(f"{label} checkpoint: missing={incompatible.missing_keys}, "
          f"unexpected={incompatible.unexpected_keys}")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{label}: strict checkpoint load was not complete")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _begin_measure(device: torch.device) -> float:
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return perf_counter()


def _end_measure(start: float, device: torch.device) -> tuple[float, str]:
    _sync(device)
    peak = (f"{torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB"
            if device.type == "cuda" else "n/a (CPU)")
    return perf_counter() - start, peak


def _mapped_names(reference: nn.Module) -> dict[str, str]:
    return {next(iter(convert_nhwc_nat_state_dict({name: parameter}))): name
            for name, parameter in reference.named_parameters()
            if not name.startswith("head.")}


def _linear_shape(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Remove only the two singleton dimensions introduced for pointwise convs."""
    if tensor.ndim == 4 and tensor.shape[-2:] == (1, 1) and target.ndim == 2:
        return tensor[:, :, 0, 0]
    return tensor


def _difference(label: str, left: torch.Tensor, right: torch.Tensor,
                rtol: float, atol: float) -> tuple[float, bool]:
    diff = (left - right).abs()
    maximum = diff.max().item() if diff.numel() else 0.0
    mean = diff.mean().item() if diff.numel() else 0.0
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    passed = finite and torch.allclose(left, right, rtol=rtol, atol=atol)
    print(f"{label}: mean_abs={mean:.6e}, max_abs={maximum:.6e}, "
          f"finite={finite}, pass={passed} (rtol={rtol:g}, atol={atol:g})")
    return maximum, passed


def _parameter_comparison(label: str, left: nn.Module, right: nn.Module,
                          right_to_left: dict[str, str] | None, rtol: float,
                          atol: float) -> tuple[bool, list[tuple[float, str]]]:
    left_params, right_params = dict(left.named_parameters()), dict(right.named_parameters())
    mapping = right_to_left or {name: name for name in right_params}
    ok, worst = True, []
    print(f"\n{label} ({len(mapping)} named parameter gradients):")
    for right_name in sorted(mapping):
        left_name = mapping[right_name]
        lp, rp = left_params.get(left_name), right_params.get(right_name)
        if lp is None or rp is None:
            print(f"  {left_name} <-> {right_name}: MISSING PARAMETER")
            ok = False
            continue
        if lp.grad is None or rp.grad is None:
            missing = [side for side, grad in (("left", lp.grad), ("right", rp.grad)) if grad is None]
            print(f"  {left_name} <-> {right_name}: MISSING GRADIENT ({', '.join(missing)})")
            ok = False
            continue
        rg = _linear_shape(rp.grad, lp.grad)
        maximum, passed = _difference(f"  {left_name} <-> {right_name}", lp.grad, rg, rtol, atol)
        worst.append((maximum, f"{left_name} <-> {right_name}"))
        ok &= passed
    print("Worst parameters:")
    for maximum, name in sorted(worst, reverse=True)[:10]:
        print(f"  {maximum:.6e}  {name}")
    return ok, worst


def _optimizer_deltas(model: nn.Module, lr: float) -> dict[str, torch.Tensor]:
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    torch.optim.SGD(model.parameters(), lr=lr).step()
    return {name: parameter.detach() - before[name] for name, parameter in model.named_parameters()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", "--encoder", choices=VARIANTS, default="nat_mini")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="local official/checkpoint state-dict path")
    source.add_argument("--pretrained", action="store_true", help="download the official ImageNet checkpoint")
    parser.add_argument("--tile-size", type=int, default=385)
    parser.add_argument("--input-size", type=int, default=481)
    parser.add_argument("--tile-cache", type=Path)
    parser.add_argument("--fresh-tile-statistics", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--forward-rtol", type=float, default=2e-4)
    parser.add_argument("--forward-atol", type=float, default=2e-5)
    parser.add_argument("--image-grad-rtol", type=float, default=5e-4)
    parser.add_argument("--image-grad-atol", type=float, default=5e-5)
    parser.add_argument("--param-grad-rtol", type=float, default=1e-3)
    parser.add_argument("--param-grad-atol", type=float, default=1e-5)
    parser.add_argument("--update-rtol", type=float, default=1e-3)
    parser.add_argument("--update-atol", type=float, default=1e-7)
    args = parser.parse_args()

    if args.input_size <= args.tile_size:
        parser.error("--input-size must exceed --tile-size (multi-tile traversal is required)")
    if args.fresh_tile_statistics:
        if args.tile_cache is None:
            args.tile_cache = Path.cwd() / (
                f"{args.variant}_tile_cache_1_3_{args.tile_size}_{args.tile_size}"
            )
        if args.tile_cache.exists():
            args.tile_cache.unlink()
            print(f"Removed tile cache for fresh statistics: {args.tile_cache}")

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    dtype = getattr(torch, args.dtype)
    print(
        f"device={device}, dtype={dtype}, encoder={args.variant}, "
        f"tile_size={args.tile_size}, input_size={args.input_size}"
    )
    # Own the source checkpoint independently of all three model instances.
    initial = _reference(args.variant)
    loaded = _checkpoint(args.checkpoint, args.pretrained, args.variant)
    if loaded is not None:
        _strict_load(initial, loaded, "checkpoint source")
    checkpoint = {name: value.detach().cpu().clone() for name, value in initial.state_dict().items()}
    del initial, loaded

    reference = _reference(args.variant)
    _strict_load(reference, checkpoint, "NHWC reference")
    feature_state = {name: value for name, value in checkpoint.items() if not name.startswith("head.")}
    full = VARIANTS[args.variant]["nchw"]()
    converted = convert_nhwc_nat_state_dict(feature_state)
    _strict_load(full, converted, "full NCHW")
    stream = StreamingNAT(
        args.variant, args.tile_size, pretrained=checkpoint, tile_cache_path=args.tile_cache,
        device=device, verbose=True, saliency=True, statistics_on_cpu=device.type == "cuda",
        normalize_on_gpu=False, mean=[0, 0, 0], std=[1, 1, 1], drop_rate=0.0,
        attn_drop_rate=0.0, drop_path_rate=0.0,
    )
    # Make the third ownership boundary and strict load explicit after conversion.
    _strict_load(stream.stream_module, converted, "streamed NCHW")
    reference.to(device=device, dtype=dtype).eval()
    full.to(device=device, dtype=dtype).eval()
    stream.to(device=device, dtype=dtype).eval()
    stream.stream_network.device, stream.stream_network.dtype = device, dtype

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    raw = torch.rand((1, 3, args.input_size, args.input_size), generator=generator,
                     device=device, dtype=dtype)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)[None, :, None, None]
    normalized = (raw - mean) / std
    ref_image = normalized.detach().clone().requires_grad_(True)
    full_image = normalized.detach().clone().requires_grad_(True)
    stream_image = normalized.detach().clone().requires_grad_(True)

    start = _begin_measure(device)
    ref_output = reference.forward_feature_map(ref_image).permute(0, 3, 1, 2)
    ref_time, ref_peak = _end_measure(start, device)
    start = _begin_measure(device)
    full_output = full(full_image)
    full_time, full_peak = _end_measure(start, device)
    start = _begin_measure(device)
    stream_output = stream(stream_image)
    stream_forward_time, stream_peak = _end_measure(start, device)

    scnn = stream.stream_network
    starts = scnn._tile_start_list(scnn._last_forward_tiles)
    rows, cols = len({y for y, _ in starts}), len({x for _, x in starts})
    valid_sizes = scnn._compute_valid_output_sizes()
    step = scnn._compute_valid_input_step(*valid_sizes)
    print("\nTraversal:")
    print(f"  tile dimensions: {tuple(int(x) for x in scnn.tile_shape[-2:])}")
    print(f"  valid input step: {tuple(int(x) for x in step)}")
    print(f"  tile rows/columns: {rows}/{cols}")
    print(f"  tile starts: {starts}")
    print(f"  output stride: {tuple(int(x) for x in scnn.output_stride.tolist())}")
    if len(starts) <= 1:
        raise RuntimeError("streaming did not genuinely traverse multiple tiles")

    upstream = torch.randn(full_output.shape, generator=generator, device=device, dtype=dtype)
    start = _begin_measure(device)
    ref_output.backward(upstream)
    ref_backward_time, ref_backward_peak = _end_measure(start, device)
    start = _begin_measure(device)
    full_output.backward(upstream)
    full_backward_time, full_backward_peak = _end_measure(start, device)
    start = _begin_measure(device)
    stream.backward_streaming(stream_image, upstream)
    stream_backward_time, stream_backward_peak = _end_measure(start, device)

    print("\nRuntime / peak CUDA memory:")
    print(f"  NHWC: forward={ref_time:.3f}s ({ref_peak}), backward={ref_backward_time:.3f}s ({ref_backward_peak})")
    print(f"  full NCHW: forward={full_time:.3f}s ({full_peak}), backward={full_backward_time:.3f}s ({full_backward_peak})")
    print(f"  streamed NCHW: forward={stream_forward_time:.3f}s ({stream_peak}), backward={stream_backward_time:.3f}s ({stream_backward_peak})")

    failures = []
    _, passed = _difference("NHWC vs full NCHW forward", ref_output, full_output,
                            args.forward_rtol, args.forward_atol)
    failures += [] if passed else ["NHWC/full forward"]
    _, passed = _difference("full NCHW vs streamed NCHW forward", full_output, stream_output,
                            args.forward_rtol, args.forward_atol)
    failures += [] if passed else ["full/stream forward"]
    stream_image_grad = scnn.saliency_map.to(device=device, dtype=dtype)
    for label, left, right in (("NHWC vs full NCHW image gradient", ref_image.grad, full_image.grad),
                               ("full NCHW vs streamed NCHW image gradient", full_image.grad, stream_image_grad)):
        if left is None or right is None:
            print(f"{label}: MISSING GRADIENT")
            failures.append(label)
        else:
            _, passed = _difference(label, left, right, args.image_grad_rtol, args.image_grad_atol)
            failures += [] if passed else [label]

    mapping = _mapped_names(reference)
    passed, _ = _parameter_comparison("NHWC vs full NCHW", reference, full, mapping,
                                      args.param_grad_rtol, args.param_grad_atol)
    failures += [] if passed else ["NHWC/full parameter gradients"]
    passed, _ = _parameter_comparison("full NCHW vs streamed NCHW", full, stream.stream_module,
                                      None, args.param_grad_rtol, args.param_grad_atol)
    failures += [] if passed else ["full/stream parameter gradients"]

    ref_delta, full_delta = _optimizer_deltas(reference, args.learning_rate), _optimizer_deltas(full, args.learning_rate)
    stream_delta = _optimizer_deltas(stream.stream_module, args.learning_rate)
    print("\nOptimizer update-delta differences:")
    for label, left, right, names in (("NHWC vs full NCHW", ref_delta, full_delta, mapping),
                                      ("full NCHW vs streamed NCHW", full_delta, stream_delta,
                                       {name: name for name in stream_delta})):
        for right_name, left_name in sorted(names.items()):
            right_value = _linear_shape(right[right_name], left[left_name])
            _, passed = _difference(f"  {label}: {left_name} <-> {right_name}", left[left_name],
                                    right_value, args.update_rtol, args.update_atol)
            failures += [] if passed else [f"{label} optimizer update {right_name}"]

    if failures:
        print("\nFAILED tolerances: " + "; ".join(failures))
        raise SystemExit(1)
    print("\nAll configured comparisons passed.")


if __name__ == "__main__":
    main()
