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

::: lightstream.core.scnn
