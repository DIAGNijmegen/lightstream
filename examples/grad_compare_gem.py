from __future__ import annotations

import copy

import torch
import torch.nn as nn

from lightstream.core.reducer import AttentionGeMReducer
from lightstream.core.scnn.scnn import StreamingCNN


class TinyAttentionGeMNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.attn_logits = nn.Conv2d(32, 1, kernel_size=1, bias=False)
        self.value = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.reducer = AttentionGeMReducer(r_init=3.0, learnable_r=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        logits = self.attn_logits(feat)
        value = self.value(feat)
        return value, self.reducer(value, logits)


def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
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

    input_tensor = torch.rand((1, 3, 8, 8), device=device, dtype=dtype)

    base_model = TinyAttentionGeMNet().to(device=device, dtype=dtype)
    normal_model = copy.deepcopy(base_model).to(device=device, dtype=dtype)
    streamed_model = copy.deepcopy(base_model).to(device=device, dtype=dtype)

    tile_size = 4
    stream_network = StreamingCNN(
        streamed_model,
        tile_shape=(1, 3, tile_size, tile_size),
        normalize_on_gpu=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        copy_to_gpu=True,
        saliency=False,
        dtype=dtype,
    )
    stream_network.device = device
    stream_network.dtype = dtype
    stream_network.mean = stream_network.mean.to(device=device, dtype=dtype)
    stream_network.std = stream_network.std.to(device=device, dtype=dtype)

    criterion = nn.BCELoss()
    target = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)

    print("===== forward =====")
    _zero_grads(stream_network.stream_module)
    stream_value_map, stream_output = stream_network(input_tensor)
    stream_output.requires_grad = True
    stream_prob = torch.sigmoid(stream_output)

    _zero_grads(normal_model)
    normal_value_map, normal_output = normal_model(input_tensor)
    normal_prob = torch.sigmoid(normal_output)

    print(f"stream_value_map shape: {tuple(stream_value_map.shape)}")
    print(f"normal_value_map shape: {tuple(normal_value_map.shape)}")
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
    _zero_grads(stream_network.stream_module)
    stream_loss.backward()
    stream_network.backward(input_tensor, (torch.zeros_like(stream_value_map), stream_output.grad))

    _zero_grads(normal_model)
    normal_loss.backward()

    print("===== grad compare =====")
    stream_params = dict(stream_network.stream_module.named_parameters())
    normal_params = dict(normal_model.named_parameters())

    _compare_grad(
        "trunk.weight",
        stream_params["trunk.weight"].grad,
        normal_params["trunk.weight"].grad,
    )
    _compare_grad(
        "attn_logits.weight",
        stream_params["attn_logits.weight"].grad,
        normal_params["attn_logits.weight"].grad,
    )
    _compare_grad(
        "value.0.weight",
        stream_params["value.0.weight"].grad,
        normal_params["value.0.weight"].grad,
    )


if __name__ == "__main__":
    main()
