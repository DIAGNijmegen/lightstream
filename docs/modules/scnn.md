# Streaming CNN layers

`lightstream.core.scnn` is the documented public module for streaming CNN layer
building blocks. Import channel layer normalization and layer scale helpers from
this package instead of private implementation modules:

```python
from lightstream.core.scnn import ChannelLayerNorm, LayerScale, StreamingChannelLayerNorm
```

## LayerScale for discoverable learned scaling

Use `LayerScale` when model code needs a learned multiplicative scale that the
streaming converter can discover and replace with `StreamingLayerScale`. For
example, replace a raw parameter such as this:

```python
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gamma
```

with a module-based scale:

```python
from lightstream.core.scnn import LayerScale

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_scale = LayerScale(shape=1, init_value=0.0)

    def forward(self, x):
        return self.gamma_scale(x)
```

`LayerScale` is also useful in residual branches where the learnable `gamma`
starts at zero:

```python
from lightstream.core.scnn import LayerScale

class ResidualBlock(torch.nn.Module):
    def __init__(self, branch):
        super().__init__()
        self.branch = branch
        self.gamma_scale = LayerScale(shape=1, init_value=0.0)

    def forward(self, residual):
        branch = self.branch(residual)
        out = residual + self.gamma_scale(branch)
        return out
```

Supported scale shapes follow PyTorch broadcasting semantics. Common choices
include scalar `(1,)`, channel-wise `(1, C, 1, 1)`, or any shape broadcastable to
the input tensor.

Raw `nn.Parameter` multiplications inside arbitrary `forward` code are not
represented as child modules, so the streaming converter cannot discover or
replace them automatically. `LayerScale` makes the operation explicit in the
module tree, allowing `StreamingCNN` conversion to replace it with
`StreamingLayerScale` while preserving the `scale` state-dict key.

## Saliency and input gradients

Input-gradient collection is optional and is separate from parameter-gradient
training. Choose the mode when constructing the streaming model:

```python
# Normal training: no input saliency overhead.
network = StreamingSSHR(
    "resnet18",
    tile_size=3072,
    saliency=False,  # Default
)

# Generate an input saliency map.
network = StreamingSSHR(
    "resnet18",
    tile_size=3072,
    saliency=True,
)
```

| Use case | `saliency` | Assembly diagnostics |
| --- | --- | --- |
| Production training | `False` | Off |
| Production saliency generation | `True` | Off |
| Basic gradient comparison | `True` | Off |
| Detailed saliency debugging | `True` | On |
| Parameter-only comparison | `False` | Off |

### Normal production training

Construct the model with `saliency=False`, which is the default. Parameter
gradients continue to work: this setting disables only collection of the input
gradient. No input-gradient hook or saliency map is required, avoiding their
memory and backward-processing overhead.

### Production saliency generation

Construct the model with `saliency=True`. In this mode,
`StreamingCNN.backward` gathers input gradients and assembles `saliency_map`.
This incurs additional memory use and backward-processing cost. Configure the
mode at construction time; do not mutate `gather_input_gradient` after the
constructor has installed the associated hooks.

### Saliency assembly diagnostics

Assembly diagnostics are disabled by default and should be enabled only with
the comparison scripts' `--diagnose-saliency-assembly` option. The `raw` and
`production` candidate maps must pass comparison with the non-streaming input
gradient. The `grad_lost` and `ownership` candidates represent counterfactual
legacy assembly stages and are expected to diverge; they characterize why
those older assembly approaches are unsuitable rather than defining additional
success criteria. Diagnostic candidate maps and write-count maps are not
created during ordinary training.

Full-resolution saliency maps—and, especially, diagnostic candidate maps—scale
with the complete NCHW input tensor. A float64 RGB 4608 × 4608 map is
approximately 486 MiB, and detailed diagnostics may allocate several such
maps. Use detailed diagnostics only when that memory cost is acceptable.

::: lightstream.core.scnn
