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

## Output stride and input-alignment metadata

After `StreamingCNN` has been configured, it exposes four read-only metadata
attributes:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `output_strides` | `tuple[tuple[int, int], ...]` | Spatial `(height, width)` input-relative stride for every flattened statistics output, in flattening order. |
| `required_input_alignment` | `tuple[int, int]` | Input `(height, width)` lattice required by all internal strided convolutions and pooling operations. |
| `output_metadata` | `tuple[Mapping, ...]` | Diagnostic entries pairing each flattened output's stable `path` with its `stride` and setup-time `shape`. Both the tuple and entries are immutable. |
| `named_output_strides` | `Mapping[str, tuple[int, int]]` | Convenient immutable lookup from a stable output path directly to its spatial stride. |

The existing `output_stride` remains the primary output's stride and keeps its
legacy tensor form (`[1, height_stride, width_stride]`). For a single-output
model, therefore, `tuple(scnn.output_stride[-2:])` describes the same spatial
stride as `scnn.output_strides[0]`.

### Input alignment is not output stride

`required_input_alignment` describes where internal sampling grids line up; it
does not describe the resolution returned by the model. For example, an
encoder feature at stride 8 followed by average pooling at stride 64 requires
input alignment `(512, 512)`, even if the primary returned output has stride 8
or is later upsampled to stride 1.

Padding is deliberately left to the calling data pipeline. Read the property
and independently round the image height and width upward according to your
application's existing padding policy:

```python
align_h, align_w = scnn.required_input_alignment
height, width = image.shape[-2:]

padded_height = ((height + align_h - 1) // align_h) * align_h
padded_width = ((width + align_w - 1) // align_w) * align_w
# Apply your normal image/target padding to padded_height × padded_width.
```

To fail early instead, either call the validator directly or opt into it on a
forward pass:

```python
scnn.validate_input_alignment(image)

# Equivalent validation immediately before tiled execution:
prediction = scnn(image, validate_input_alignment=True)
```

An alignment error reports the received spatial shape, the required alignment,
and the next valid padded shape. Validation is optional; the streaming merge
checks remain active as defensive assertions.

### Selecting the stride for an output

Use `output_metadata` when a model returns nested dictionaries, lists, or
tuples. Paths use dictionary names and bracketed sequence indices, and entries
stay in the same order as `output_strides`:

```python
for index, metadata in enumerate(scnn.output_metadata):
    assert metadata["stride"] == scnn.output_strides[index]
    print(metadata["path"], metadata["stride"], metadata["shape"])

# Example paths from a structured model:
# loss_components[0][0] (8, 8) (1, 4, 64, 64)
# loss_components[0][1] (8, 8) (1, 4, 64, 64)
# p_fused              (1, 1) (1, 4, 512, 512)
```

For frequent path-based stride lookups, use `named_output_strides` directly
instead of rebuilding a dictionary from `output_metadata`:

```python
fused_stride = scnn.named_output_strides["p_fused"]
loss_stride = scnn.named_output_strides["loss_components[0][0]"]
```

When cropping a stride-1 prediction after application-managed padding, crop it
to the original height and width. For a native stride-4 or stride-8 output,
first find that output by `path` in `output_metadata`, then translate the
original extent using that entry's `stride`. Never use
`required_input_alignment` (for example, 512) as an output-cropping stride.

::: lightstream.core.scnn
