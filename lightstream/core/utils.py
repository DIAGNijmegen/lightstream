import torch.nn as nn

# Layernorm could potentially be trained? depending on the configuration?
# But must also be made streamable with custom function like streamingconv

NORM_LAYERS = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
]

def freeze_normalization_layers(model: nn.Module, norm_types: tuple[type, ...] = tuple(NORM_LAYERS)) -> None:
    """
    Freezes all normalization layers in a model:
    - Disables gradient computation
    - Sets the module to eval mode
    - Disables running stats if applicable

    Args:
        model: The PyTorch model to modify.
        norm_types: A tuple of normalization layer classes to freeze.
    """
    for module in model.modules():
        if isinstance(module, norm_types):
            module.eval()
            module.requires_grad_(False)

def unfreeze_streaming_network(model: nn.Module, norm_types: tuple[type, ...] = tuple(NORM_LAYERS)) -> None:
    for module in model.modules():
        if isinstance(module, norm_types):
            module.eval()
            module.requires_grad_(False)
        else:
            module.requires_grad_(True)