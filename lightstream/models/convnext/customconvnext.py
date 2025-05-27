import torch
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)
from torchvision.models.convnext import CNBlock
from types import MethodType


def patched_forward(self, input: torch.Tensor) -> torch.Tensor:
    """
    turn layer scale to (b)float16 when using autocast. Lowers memory, and should be stable under bfloat16
    """
    out = self.block(input)
    if self.layer_scale.dtype != out.dtype or self.layer_scale.device != out.device:
        scale = self.layer_scale.to(dtype=out.dtype, device=out.device)
    else:
        scale = self.layer_scale
    #out = scale * out
    #out = self.stochastic_depth(out)
    out = out + input
    return out


# Patch all CNBlock forward methods recursively
def patch_cnblock_forwards(module):
    for child in module.children():
        if isinstance(child, CNBlock):
            child.forward = MethodType(patched_forward, child)
        else:
            patch_cnblock_forwards(child)


# Mapping from variant string to constructor and weights
CONVNEXT_VARIANTS = {
    "tiny": convnext_tiny,
    "small": convnext_small,
    "base": convnext_base,
    "large": convnext_large,
}


class CustomConvNeXt(torch.nn.Module):
    def __init__(self, variant="tiny", weights="DEFAULT", progress=True):
        super().__init__()
        variant = variant.lower()
        if variant not in CONVNEXT_VARIANTS:
            raise ValueError(f"Unsupported variant '{variant}'. Choose from {list(CONVNEXT_VARIANTS.keys())}")

        constructor = CONVNEXT_VARIANTS[variant]
        base_model = constructor(weights=weights, progress=progress)

        # Unpack and register submodules directly
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier


        patch_cnblock_forwards(self.features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def custom_convnext_tiny(weights: str = "DEFAULT", progress: bool = True) -> CustomConvNeXt:
    return CustomConvNeXt("tiny", weights, progress)


def custom_convnext_small(weights: str = "DEFAULT", progress: bool = True) -> CustomConvNeXt:
    return CustomConvNeXt("small", weights, progress)


def custom_convnext_base(weights: str = "DEFAULT", progress: bool = True) -> CustomConvNeXt:
    return CustomConvNeXt("base", weights, progress)


def custom_convnext_large(weights: str = "DEFAULT", progress: bool = True) -> CustomConvNeXt:
    return CustomConvNeXt("large", weights, progress)


# --- Comparison test ---
if __name__ == "__main__":
    from torch.amp import autocast

    device = "cuda" if torch.cuda.is_available() else "cpu"

    orig_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).to(device).eval()

    patched_model = custom_convnext_tiny(weights="DEFAULT").to(device).eval()

    x = torch.randn(1, 3, 224, 224).to(device)

    with autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            out_orig = orig_model(x)
            out_patch = patched_model(x)

    print(f"\nOriginal output dtype: {out_orig.dtype}")
    print(f"Patched  output dtype: {out_patch.dtype}")

    diff = (out_orig - out_patch).abs().mean().item()
    print(f"Mean absolute difference: {diff:.6f}")
