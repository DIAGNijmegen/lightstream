from __future__ import annotations

from typing import Iterable, Sequence
import argparse

import torch
import torch.nn as nn
from time import time
from lightstream.models.segment.streamingwss import StreamingWSS


GRAD_MODES = ("red123_only", "red4_only", "red1_only", "all")

MODE_ACTIVE_OUTPUTS: dict[str, tuple[int, ...]] = {
    "red123_only": (0, 1, 2),
    "red4_only": (3,),
    "red1_only": (0,),
    "all": (0, 1, 2, 3),
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


def _selected_param_names(model: nn.Module) -> list[str]:
    selected: list[str] = []
    for name, param in model.named_parameters():
        is_decoder_or_attention = name.startswith(
            ("decoder1.", "decoder2.", "decoder3.", "att_1.", "att_2.", "att_3.")
        )
        is_backbone_conv_weight = (
            name.startswith("backbone.") and name.endswith("weight") and param.ndim == 4
        )
        if is_decoder_or_attention or is_backbone_conv_weight:
            selected.append(name)
    return selected


def _grad_or_zeros(
    grads: dict[str, torch.Tensor], params: dict[str, torch.nn.Parameter], name: str
) -> torch.Tensor:
    grad = grads.get(name)
    if grad is not None:
        return grad
    return torch.zeros_like(params[name], memory_format=torch.preserve_format).detach()


def _compare_selected_grads(
    stream_module: nn.Module,
    stream_grads: dict[str, torch.Tensor],
    normal_net: nn.Module,
    normal_grads: dict[str, torch.Tensor],
) -> None:
    stream_params = dict(stream_module.named_parameters())
    normal_params = dict(normal_net.named_parameters())
    selected_names = [name for name in _selected_param_names(normal_net) if name in stream_params]
    if not selected_names:
        print("No selected gradients found to compare.")
        return

    print(f"\nSelected gradient stats ({len(selected_names)} parameters):")
    for name in selected_names:
        stream_grad = _grad_or_zeros(stream_grads, stream_params, name)
        normal_grad = _grad_or_zeros(normal_grads, normal_params, name)
        diff = (stream_grad - normal_grad).abs()
        print(
            f"{name}: "
            f"stream mean abs={stream_grad.abs().mean().item():.6e}, "
            f"normal mean abs={normal_grad.abs().mean().item():.6e}, "
            f"mean abs diff={diff.mean().item():.6e}, "
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


def _freeze_batchnorm(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.BatchNorm2d):
            submodule.eval()
            for param in submodule.parameters():
                param.requires_grad = False


def _build_dummy_mask(size: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing="ij",
    )
    center = size // 2
    radius = max(8, size // 3)
    return ((yy - center).abs() + (xx - center).abs()) <= radius


def _base_output_grads(
    outputs: Sequence[torch.Tensor], target: torch.Tensor, criterion: nn.Module
) -> tuple[torch.Tensor, ...]:
    """Return the current loss-derived upstream gradient for every reducer output."""
    grads = torch.autograd.grad(
        sum(criterion(torch.sigmoid(torch.mean(output)), target) for output in outputs),
        tuple(outputs),
        retain_graph=True,
    )
    return tuple(grad.detach().clone() for grad in grads)


def _mode_output_grads(
    outputs: Sequence[torch.Tensor], mode: str, target: torch.Tensor, criterion: nn.Module
) -> tuple[torch.Tensor, ...]:
    base_grads = _base_output_grads(outputs, target, criterion)
    active_outputs = set(MODE_ACTIVE_OUTPUTS[mode])
    return tuple(
        base_grad if output_idx in active_outputs else torch.zeros_like(base_grad)
        for output_idx, base_grad in enumerate(base_grads)
    )


def _run_mode(args: argparse.Namespace, mode: str, img: torch.Tensor, mask: torch.Tensor) -> None:
    device = img.device
    dtype = img.dtype
    target = torch.tensor(50.0, device=device, dtype=dtype)
    criterion = torch.nn.MSELoss()

    print("\n" + "=" * 80)
    print(f"Gradient comparison mode: {mode}")
    print("=" * 80)

    network = StreamingWSS(
        "resnet18",
        args.tile_size,
        additional_modules=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
        saliency=args.input_grad,
    ).to(device=device, dtype=dtype)
    network.stream_network.device = device
    network.stream_network.dtype = dtype
    network.stream_network.mean = network.stream_network.mean.to(device=device, dtype=dtype)
    network.stream_network.std = network.stream_network.std.to(device=device, dtype=dtype)

    # Valid StreamingCNN debug information
    print("output_spec:", network.stream_network._output_spec)
    print(
        "output_stride_per_output:",
        [tuple(int(x) for x in s.tolist()) for s in network.stream_network._output_stride_per_output],
    )

    _freeze_batchnorm(network.stream_network.stream_module)

    _zero_grads(network.stream_network.stream_module.parameters())
    start_streaming_forward = time()
    stream_outputs = network(img, mask=mask)
    end_streaming_forward = time() - start_streaming_forward
    if not isinstance(stream_outputs, tuple) or len(stream_outputs) != 4:
        output_count = len(stream_outputs) if hasattr(stream_outputs, "__len__") else "unknown"
        raise RuntimeError(
            f"Expected 4 reducer outputs, got {type(stream_outputs)} with length {output_count}"
        )
    for output in stream_outputs:
        output.requires_grad = True

    output_grads = _mode_output_grads(stream_outputs, mode, target, criterion)
    print(
        "upstream grad mean abs per output:",
        [f"red{idx + 1}={grad.abs().mean().item():.6e}" for idx, grad in enumerate(output_grads)],
    )

    start_streaming_backward = time()
    network.stream_network.backward(img, output_grads, mask=mask)
    end_streaming_backward = time() - start_streaming_backward

    streaming_param_grads = _gather_param_grads(network.stream_network.stream_module)

    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    _freeze_batchnorm(normal_net)
    _zero_grads(normal_net.parameters())
    img_normal = img.detach().clone().requires_grad_(args.input_grad)

    start_normal_forward = time()
    normal_outputs = normal_net(img_normal, mask=mask)
    end_normal_forward = time() - start_normal_forward

    for idx, (stream_out, normal_out) in enumerate(zip(stream_outputs, normal_outputs), start=1):
        diff = (stream_out - normal_out).abs()
        print(f"red{idx} forward output sum/max diff: {diff.sum().item()}, {diff.max().item()}")

    start_normal_backward = time()
    torch.autograd.backward(normal_outputs, tuple(grad.detach().clone() for grad in output_grads))
    end_normal_backward = time() - start_normal_backward
    normal_param_grads = _gather_param_grads(normal_net)

    if args.input_grad:
        if img_normal.grad is None:
            print("Input gradient comparison skipped: non-streaming input gradient is missing.")
        elif not hasattr(network.stream_network, "saliency_map") or network.stream_network.saliency_map is None:
            print("Input gradient comparison skipped: streaming saliency map is missing.")
        else:
            stream_input_grad = network.stream_network.saliency_map[0].to(device=img_normal.grad.device)
            input_grad_diff = (img_normal.grad.detach() - stream_input_grad).abs()
            print(
                "Input gradient stats: "
                f"stream mean abs={stream_input_grad.abs().mean().item():.6e}, "
                f"normal mean abs={img_normal.grad.detach().abs().mean().item():.6e}, "
                f"mean abs diff={input_grad_diff.mean().item():.6e}, "
                f"max abs diff={input_grad_diff.max().item():.6e}"
            )

    _compare_selected_grads(
        network.stream_network.stream_module,
        streaming_param_grads,
        normal_net,
        normal_param_grads,
    )

    print("\nTimings")
    print("Time spent in streaming forward:", end_streaming_forward)
    print("Time spent in streaming backward:", end_streaming_backward)
    print("Time spent in normal forward:", end_normal_forward)
    print("Time spent in normal backward:", end_normal_backward)

    print("\nTime difference forward streaming vs normal", end_streaming_forward - end_normal_forward)
    print("Time difference backward streaming vs normal", end_streaming_backward - end_normal_backward)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare streaming vs non-streaming backward gradients for WSS.")
    parser.add_argument("--dtype", default="float64", help="float16, float32, or float64")
    parser.add_argument("--tile-size", type=int, default=2560)
    parser.add_argument("--input-size", type=int, default=5120)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(GRAD_MODES),
        choices=GRAD_MODES,
        help="Gradient routing modes to run. Defaults to every mode.",
    )
    parser.add_argument(
        "--no-input-grad",
        dest="input_grad",
        action="store_false",
        help="Disable streaming saliency/input-gradient gathering and skip input-gradient comparison.",
    )
    parser.set_defaults(input_grad=True)

    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _parse_dtype(args.dtype)

    img = torch.rand((1, 3, args.input_size, args.input_size), device=device, dtype=dtype)
    mask = _build_dummy_mask(args.input_size, device=device)

    print(f"device={device}, dtype={dtype}, tile_size={args.tile_size}, input_size={args.input_size}")
    print("modes:", ", ".join(args.modes))
    for mode in args.modes:
        _run_mode(args, mode, img, mask)


if __name__ == "__main__":
    main()