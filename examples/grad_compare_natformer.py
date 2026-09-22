"""Compare NHWC, NCHW, and streamed NAT feature-map gradients.

Example
-------
Compare the hosted NHWC and NCHW ImageNet checkpoints::

    python examples/grad_compare_natformer.py --variant nat_mini

To deliberately regenerate statistics, add ``--fresh-tile-statistics``.  Without
that flag ``--tile-cache PATH`` loads a compatible existing cache (and creates
and saves it when it does not yet exist).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from lightstream.models.nat.nat import (
    nat_base,
    nat_mini,
    nat_nano,
    nat_pico,
    nat_small,
    nat_tiny,
)
from lightstream.models.nat.streaming import StreamingNAT

NHWC_MODEL_CHOICES = {
    "nat_mini": nat_mini,
    "nat_tiny": nat_tiny,
    "nat_small": nat_small,
    "nat_base": nat_base,
    "nat_nano": nat_nano,
    "nat_pico": nat_pico,
}
PRETRAINED_CHOICES = ("nat_mini", "nat_tiny", "nat_small", "nat_base")


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
    peak = (
        f"{torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB"
        if device.type == "cuda"
        else "n/a (CPU)"
    )
    return perf_counter() - start, peak


def _mapped_names(reference: nn.Module) -> dict[str, str]:
    """Map only the representation wrappers used by the NCHW implementation."""
    mapped = {}
    for name, parameter in reference.named_parameters():
        if name.startswith("head."):
            continue
        parts = name.split(".")
        if "attn" in parts:
            parts.insert(parts.index("attn") + 1, "attention")
        for index, part in tuple(enumerate(parts)):
            if part in {"norm", "norm1", "norm2"} and index + 1 < len(parts):
                if parts[index + 1] in {"weight", "bias"}:
                    parts.insert(index + 1, "norm")
                break
        if name.endswith(("gamma1", "gamma2")):
            parts.append("weight")
        target = ".".join(parts)
        if target in mapped:
            raise RuntimeError(f"duplicate NCHW counterpart for {name!r}")
        mapped[target] = name
    return mapped


def _linear_shape(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Adapt pointwise-convolution and NCHW layer-scale parameter shapes."""
    if tensor.ndim == 4 and tensor.shape[-2:] == (1, 1) and target.ndim == 2:
        return tensor[:, :, 0, 0]
    if tensor.ndim == 4 and tensor.shape[:1] == (1,) and target.ndim == 1:
        return tensor[0, :, 0, 0]
    return tensor


def _difference(
    label: str, left: torch.Tensor, right: torch.Tensor, rtol: float, atol: float
) -> tuple[float, bool]:
    diff = (left - right).abs()
    maximum = diff.max().item() if diff.numel() else 0.0
    mean = diff.mean().item() if diff.numel() else 0.0
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    passed = finite and torch.allclose(left, right, rtol=rtol, atol=atol)
    print(
        f"{label}: mean_abs={mean:.6e}, max_abs={maximum:.6e}, "
        f"finite={finite}, pass={passed} (rtol={rtol:g}, atol={atol:g})"
    )
    return maximum, passed


def _parameter_comparison(
    label: str,
    left: nn.Module,
    right: nn.Module,
    right_to_left: dict[str, str] | None,
    rtol: float,
    atol: float,
) -> tuple[bool, list[tuple[float, str]]]:
    left_params, right_params = dict(left.named_parameters()), dict(
        right.named_parameters()
    )
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
            missing = [
                side
                for side, grad in (("left", lp.grad), ("right", rp.grad))
                if grad is None
            ]
            print(
                f"  {left_name} <-> {right_name}: MISSING GRADIENT ({', '.join(missing)})"
            )
            ok = False
            continue
        rg = _linear_shape(rp.grad, lp.grad)
        maximum, passed = _difference(
            f"  {left_name} <-> {right_name}", lp.grad, rg, rtol, atol
        )
        worst.append((maximum, f"{left_name} <-> {right_name}"))
        ok &= passed
    print("Worst parameters:")
    for maximum, name in sorted(worst, reverse=True)[:10]:
        print(f"  {maximum:.6e}  {name}")
    return ok, worst


def _optimizer_deltas(model: nn.Module, lr: float) -> dict[str, torch.Tensor]:
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    torch.optim.SGD(model.parameters(), lr=lr).step()
    return {
        name: parameter.detach() - before[name]
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant",
        "--encoder",
        default="nat_mini",
        choices=PRETRAINED_CHOICES,
    )
    parser.add_argument("--tile-size", type=int, default=4680)
    parser.add_argument("--input-size", type=int, default=5120)
    parser.add_argument("--tile-cache", type=Path)
    parser.add_argument("--fresh-tile-statistics", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "float64"), default="float64"
    )
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

    nhwc_factory = NHWC_MODEL_CHOICES[args.variant]
    nchw_factory = StreamingNAT.get_model_choices()[args.variant]

    if args.input_size <= args.tile_size:
        parser.error(
            "--input-size must exceed --tile-size (multi-tile traversal is required)"
        )
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
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dtype = getattr(torch, args.dtype)
    print(
        f"device={device}, dtype={dtype}, encoder={args.variant}, "
        f"tile_size={args.tile_size}, input_size={args.input_size}"
    )
    reference = nhwc_factory(pretrained=True)
    full = nchw_factory(pretrained=True)
    stream = StreamingNAT(
        args.variant,
        args.tile_size,
        pretrained=True,
        tile_cache_path=args.tile_cache,
        device=device,
        verbose=True,
        saliency=True,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )
    scnn = stream.stream_network
    streamed_model = scnn.stream_module
    full_state = full.state_dict()
    streamed_state = streamed_model.state_dict()
    if full_state.keys() != streamed_state.keys() or any(
        not torch.equal(
            value.detach().cpu(),
            streamed_state[name].detach().cpu(),
        )
        for name, value in full_state.items()
    ):
        raise RuntimeError("full and streamed NCHW checkpoints are not identical")
    reference.to(device=device, dtype=dtype).eval()
    full.to(device=device, dtype=dtype).eval()
    stream.to(device=device, dtype=dtype).eval()
    scnn.device, scnn.dtype = device, dtype

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    raw = torch.rand(
        (1, 3, args.input_size, args.input_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)[
        None, :, None, None
    ]
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)[
        None, :, None, None
    ]
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

    upstream = torch.randn(
        full_output.shape, generator=generator, device=device, dtype=dtype
    )
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
    print(
        f"  NHWC: forward={ref_time:.3f}s ({ref_peak}), backward={ref_backward_time:.3f}s ({ref_backward_peak})"
    )
    print(
        f"  full NCHW: forward={full_time:.3f}s ({full_peak}), backward={full_backward_time:.3f}s ({full_backward_peak})"
    )
    print(
        f"  streamed NCHW: forward={stream_forward_time:.3f}s ({stream_peak}), backward={stream_backward_time:.3f}s ({stream_backward_peak})"
    )

    failures = []
    _, passed = _difference(
        "NHWC vs full NCHW forward",
        ref_output,
        full_output,
        args.forward_rtol,
        args.forward_atol,
    )
    failures += [] if passed else ["NHWC/full forward"]
    _, passed = _difference(
        "full NCHW vs streamed NCHW forward",
        full_output,
        stream_output,
        args.forward_rtol,
        args.forward_atol,
    )
    failures += [] if passed else ["full/stream forward"]
    stream_image_grad = scnn.saliency_map.to(device=device, dtype=dtype)
    for label, left, right in (
        ("NHWC vs full NCHW image gradient", ref_image.grad, full_image.grad),
        (
            "full NCHW vs streamed NCHW image gradient",
            full_image.grad,
            stream_image_grad,
        ),
    ):
        if left is None or right is None:
            print(f"{label}: MISSING GRADIENT")
            failures.append(label)
        else:
            _, passed = _difference(
                label, left, right, args.image_grad_rtol, args.image_grad_atol
            )
            failures += [] if passed else [label]

    mapping = _mapped_names(reference)
    passed, _ = _parameter_comparison(
        "NHWC vs full NCHW",
        reference,
        full,
        mapping,
        args.param_grad_rtol,
        args.param_grad_atol,
    )
    failures += [] if passed else ["NHWC/full parameter gradients"]
    passed, _ = _parameter_comparison(
        "full NCHW vs streamed NCHW",
        full,
        streamed_model,
        None,
        args.param_grad_rtol,
        args.param_grad_atol,
    )
    failures += [] if passed else ["full/stream parameter gradients"]

    ref_delta, full_delta = _optimizer_deltas(
        reference, args.learning_rate
    ), _optimizer_deltas(full, args.learning_rate)
    stream_delta = _optimizer_deltas(streamed_model, args.learning_rate)
    print("\nOptimizer update-delta differences:")
    for label, left, right, names in (
        ("NHWC vs full NCHW", ref_delta, full_delta, mapping),
        (
            "full NCHW vs streamed NCHW",
            full_delta,
            stream_delta,
            {name: name for name in stream_delta},
        ),
    ):
        for right_name, left_name in sorted(names.items()):
            right_value = _linear_shape(right[right_name], left[left_name])
            _, passed = _difference(
                f"  {label}: {left_name} <-> {right_name}",
                left[left_name],
                right_value,
                args.update_rtol,
                args.update_atol,
            )
            failures += [] if passed else [f"{label} optimizer update {right_name}"]

    if failures:
        print("\nFAILED tolerances: " + "; ".join(failures))
        raise SystemExit(1)
    print("\nAll configured comparisons passed.")


if __name__ == "__main__":
    main()
