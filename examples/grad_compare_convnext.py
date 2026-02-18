from __future__ import annotations

import argparse
from typing import Iterable

import torch
import torch.nn as nn

from lightstream.models.convnext.convnext import StreamingConvNext


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
        print(
            f"{name}: "
            f"mean abs diff={diff.mean().item():.6e}, max abs diff={diff.max().item():.6e}, "
        )


def _compare_spatial_weight_grads(
    stream_grads: dict[str, torch.Tensor], normal_grads: dict[str, torch.Tensor]
) -> None:
    kernel_names = sorted(
        name
        for name, grad in stream_grads.items()
        if name in normal_grads and grad.ndim >= 3
    )
    if not kernel_names:
        print("No spatial kernel gradients found to compare.")
        return

    print("\nSpatial kernel gradient stats:")
    for name in kernel_names:
        stream_grad = stream_grads[name]
        normal_grad = normal_grads[name]
        diff = (stream_grad - normal_grad).abs()
        print(
            f"{name}: "
            f"stream mean abs={stream_grad.abs().mean().item():.6e}, "
            f"normal mean abs={normal_grad.abs().mean().item():.6e}, "
            f"mean abs diff={stream_grad.abs().mean().item() - normal_grad.abs().mean().item():.6e}, "
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare streaming vs non-streaming backward gradients for ConvNeXt.")
    parser.add_argument("--encoder", default="convnext_tiny_hnf")
    parser.add_argument("--dtype", default="float64", help="float16, float32, or float64")
    parser.add_argument("--tile-size", type=int, default=3200)
    parser.add_argument("--input-size", type=int, default=4800)
    parser.add_argument("--remove-last-block", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _parse_dtype(args.dtype)

    img = torch.rand((1, 3, args.input_size, args.input_size), device=device, dtype=dtype)
    target = torch.tensor(50.0, device=device, dtype=dtype)
    criterion = torch.nn.MSELoss()

    network = StreamingConvNext(
        args.encoder,
        args.tile_size,
        additional_modules=None,
        remove_last_block=args.remove_last_block,
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

    _zero_grads(network.stream_network.stream_module.parameters())
    stream_output = network(img)
    stream_output.requires_grad = True
    y_pred_streaming = torch.sigmoid(torch.mean(stream_output))
    loss = criterion(y_pred_streaming, target)
    loss.backward()
    network.stream_network.backward(img, stream_output.grad)
    streaming_param_grads = _gather_param_grads(network.stream_network.stream_module)

    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    _freeze_batchnorm(normal_net)
    _zero_grads(normal_net.parameters())
    img_normal = img.detach().clone().requires_grad_(True)
    normal_output = normal_net(img_normal)
    forward_diff = (stream_output - normal_output).abs()
    print(f"Forward output sum/max diff: {forward_diff.sum().item()}, {forward_diff.max().item()}")

    y_pred_normal = torch.sigmoid(torch.mean(normal_output))
    normal_loss = criterion(y_pred_normal, target)
    normal_loss.backward()
    normal_param_grads = _gather_param_grads(normal_net)

    if img_normal.grad is not None and network.stream_network.saliency_map is not None:
        input_grad_diff = img_normal.grad.detach().cpu().numpy() - network.stream_network.saliency_map[0].numpy()
        print(f"Input gradient max diff: {input_grad_diff.max()}")

    _compare_grads(streaming_param_grads, normal_param_grads)
    _compare_spatial_weight_grads(streaming_param_grads, normal_param_grads)


if __name__ == "__main__":
    main()
