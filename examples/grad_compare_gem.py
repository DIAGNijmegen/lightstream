from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from lightstream.core.reducer import AttentionGeMReducer
from lightstream.modules.streaming import StreamingModule


class TinyAttentionGeMNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.attn_logits = nn.Conv2d(32, 1, kernel_size=1, bias=False)
        self.value = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.reducer = AttentionGeMReducer(r_init=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        logits = self.attn_logits(feat)
        value = self.value(feat)
        return self.reducer(value, logits)


class StreamingTinyAttentionGeM(StreamingModule):
    def __init__(
        self,
        tile_size: int,
        verbose: bool = True,
        deterministic: bool = True,
        saliency: bool = False,
        copy_to_gpu: bool = False,
        statistics_on_cpu: bool = True,
        normalize_on_gpu: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        tile_cache_path: str | Path | None = None,
    ) -> None:
        stream_network = TinyAttentionGeMNet()

        if mean is None:
            mean = [0.0, 0.0, 0.0]
        if std is None:
            std = [1.0, 1.0, 1.0]

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
        )


def _zero_grads(parameters) -> None:
    for p in parameters:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _compare_grad(name: str, stream_grad: torch.Tensor, normal_grad: torch.Tensor) -> None:
    diff = (stream_grad - normal_grad).abs()
    print(f"{name}: mean abs diff={diff.mean().item():.6e}, max abs diff={diff.max().item():.6e}")


def main() -> None:
    print("===== setup =====")
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    tile_size = 32
    input_tensor = torch.rand((1, 3, 320, 320), device=device, dtype=dtype)

    network = StreamingTinyAttentionGeM(
        tile_size,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
        saliency=False,
    ).to(device=device, dtype=dtype)
    network.stream_network.device = device
    network.stream_network.dtype = dtype
    network.stream_network.mean = network.stream_network.mean.to(device=device, dtype=dtype)
    network.stream_network.std = network.stream_network.std.to(device=device, dtype=dtype)

    normal_model = copy.deepcopy(network.stream_network.stream_module).to(device=device, dtype=dtype)

    criterion = nn.BCELoss()
    target = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)

    print("===== forward =====")
    _zero_grads(network.stream_network.stream_module.parameters())
    stream_output = network(input_tensor)
    stream_output.requires_grad = True
    stream_prob = torch.sigmoid(stream_output)

    _zero_grads(normal_model.parameters())
    normal_output = normal_model(input_tensor)
    normal_prob = torch.sigmoid(normal_output)

    print(f"stream_output: {stream_output.detach().cpu().flatten().tolist()}")
    print(f"normal_output: {normal_output.detach().cpu().flatten().tolist()}")
    print(f"stream_prob:   {stream_prob.detach().cpu().flatten().tolist()}")
    print(f"normal_prob:   {normal_prob.detach().cpu().flatten().tolist()}")

    forward_diff = (stream_output - normal_output).abs()
    prob_diff = (stream_prob - normal_prob).abs()
    print(f"Forward output sum/max diff: {forward_diff.sum().item()}, {forward_diff.max().item()}")
    print(f"Probability output sum/max diff: {prob_diff.sum().item()}, {prob_diff.max().item()}")

    print("===== loss =====")
    stream_loss = criterion(stream_prob, target)
    normal_loss = criterion(normal_prob, target)
    print(f"stream_loss: {stream_loss.item():.12f}")
    print(f"normal_loss: {normal_loss.item():.12f}")
    loss_diff = (stream_loss.detach() - normal_loss.detach()).abs()
    print(f"Loss abs diff: {loss_diff.item():.6e}")

    print("===== backward =====")
    _zero_grads(network.stream_network.stream_module.parameters())
    stream_loss.backward()
    network.stream_network.backward(input_tensor, stream_output.grad)

    _zero_grads(normal_model.parameters())
    normal_loss.backward()

    print("===== grad compare =====")
    stream_params = dict(network.stream_network.stream_module.named_parameters())
    normal_params = dict(normal_model.named_parameters())

    _compare_grad("trunk.weight", stream_params["trunk.weight"].grad, normal_params["trunk.weight"].grad)
    _compare_grad(
        "attn_logits.weight", stream_params["attn_logits.weight"].grad, normal_params["attn_logits.weight"].grad
    )
    _compare_grad("value.0.weight", stream_params["value.0.weight"].grad, normal_params["value.0.weight"].grad)


if __name__ == "__main__":
    main()
