from __future__ import annotations

from typing import Callable, Dict, Tuple
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
)


def get_resnet_model_choices() -> dict[str, Callable[..., nn.Module]]:
    return {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
    }

_RESNET_WEIGHTS = {
    "resnet18": ResNet18_Weights,
    "resnet34": ResNet34_Weights,
    "resnet50": ResNet50_Weights,
    "resnet101": ResNet101_Weights,
}

class ResNetStages(nn.Module):
    def __init__(self, resnet: nn.Module, include_layer4: bool = False):
        super().__init__()
        self.m = resnet
        self.include_layer4 = include_layer4

        required = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"]
        if include_layer4:
            required.append("layer4")
        missing = [name for name in required if not hasattr(resnet, name)]
        if missing:
            raise TypeError(f"Expected a torchvision ResNet-like model. Missing: {missing}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # stem
        x = self.m.conv1(x)
        x = self.m.bn1(x)
        x = self.m.relu(x)
        x = self.m.maxpool(x)

        # stages
        x1 = self.m.layer1(x)
        x2 = self.m.layer2(x1)
        x3 = self.m.layer3(x2)

        if self.include_layer4:
            x4 = self.m.layer4(x3)
            return x1, x2, x3, x4

        return x1, x2, x3

def make_resnet_backbone(
    encoder: str,
    *,
    weights: str | None = "default",
    include_layer4: bool = False,
) -> tuple[ResNetStages, Dict[str, int]]:
    """
    Create a torchvision ResNet backbone that returns intermediate stage outputs.

    Args:
        encoder: one of {"resnet18","resnet34","resnet50","resnet101"}
        weights:
            - "default": use torchvision's official DEFAULT weights enum
            - None: random initialization
        include_layer4: whether to also return layer4 (x4)

    Returns:
        backbone: ResNetStages module returning (x1, x2, x3[, x4])
        channels: dict of output channels for each returned stage
    """
    builders = get_resnet_model_choices()

    if encoder not in builders:
        raise ValueError(f"Unknown encoder '{encoder}'. Available: {list(builders)}")

    builder = builders[encoder]

    if weights == "default":
        weight_enum = _RESNET_WEIGHTS[encoder]
        model = builder(weights=weight_enum.DEFAULT)
    elif weights is None:
        model = builder(weights=None)
    else:
        raise ValueError("weights must be 'default' or None")

    backbone = ResNetStages(model, include_layer4=include_layer4)

    # Infer output channels from the last block in each layer
    def layer_out_channels(layer: nn.Sequential) -> int:
        block = layer[-1]
        # Bottleneck has conv3; BasicBlock has conv2
        if hasattr(block, "conv3"):
            return int(block.conv3.out_channels)
        return int(block.conv2.out_channels)

    channels: Dict[str, int] = {
        "layer1": layer_out_channels(model.layer1),
        "layer2": layer_out_channels(model.layer2),
        "layer3": layer_out_channels(model.layer3),
    }
    if include_layer4:
        channels["layer4"] = layer_out_channels(model.layer4)

    return backbone, channels


if __name__ == "__main__":
    backbone, channels = make_resnet_backbone("resnet50", weights="default", include_layer4=False)
    print("channels:", channels)

    x = torch.randn(1, 3, 224, 224)
    x1, x2, x3 = backbone(x)
    print([t.shape for t in (x1, x2, x3)])
