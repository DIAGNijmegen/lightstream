import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from torch.cuda.amp import autocast

# Load ConvNeXt Tiny pretrained model (or your custom convnext_tiny_custom)
model = convnext_tiny(pretrained=False).cuda()
model.eval()
import pydevd

# Create a simple input tensor, batch size 1
input_tensor = torch.randn(1, 3, 224, 224).cuda()

def forward_hook(module, input, output):
    print(f"[Forward] {module.__class__.__name__}")
    print(f"  input dtype: {input[0].dtype}")
    print(f"  weight dtype: {module.weight.dtype}")
    print(f"  output dtype: {output.dtype}")

def backward_hook(module, grad_input, grad_output):
    pydevd.settrace(suspend=False, trace_only_current_thread=True)
    print(f"[Backward] {module.__class__.__name__}")
    print(f"  grad_input[0] dtype: {grad_input[0].dtype if grad_input[0] is not None else None}")
    print(f"  weight grad dtype: {module.weight.grad.dtype if module.weight.grad is not None else None}")
    print(f"  weight grad output: {grad_output[0].dtype if grad_output[0] is not None else None}")


# Register hooks on all conv2d layers
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        module.register_forward_hook(forward_hook)
        module.register_backward_hook(backward_hook)

criterion = nn.MSELoss()

# Forward and backward pass under autocast context
with autocast():
    output = model(input_tensor)
    target = torch.randn_like(output)
    loss = criterion(output, target)
loss.backward()