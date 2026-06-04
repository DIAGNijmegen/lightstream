# `lightstream.core.reducer`

This package contains the spatial reduction logic used by Lightstream in both non-streaming and streaming paths.

## Quick usage contract (important)

- Use `*Reducer` classes (for example, `MeanReducer`, `GeMReducer`) in model definitions.
- SCNN converts reducers for tiled execution via `to_streaming()`.
- `Streaming*Reducer` classes are execution implementations, **not** the primary user entrypoint.

In practice: define your model with non-streaming reducer modules, then let SCNN/runtime conversion handle the streaming class swap.

## Generalized reducer input contract

All reducer families now follow:

```python
forward(*inputs, mask=None)
```

- `*inputs` is ordered and reducer-defined.
  - Single-input reducers (for example `MeanReducer`, `GeMReducer`) still use one positional tensor.
  - Multi-input reducers must document exact positional meaning and validate it at runtime.
- `mask` remains optional and must align with the reduced spatial domain.
- Reducers should reject invalid input arity/shape early with clear errors.

### Streaming payload and replay requirements

For SCNN streaming support, reducers must preserve non-streaming semantics:

1. `to_streaming()` must pass all reducer parameters/state needed to reproduce `forward(*inputs, mask=None)` behavior.
2. Streaming replay/validation must consume inputs in identical order and apply mask semantics identically.
3. Multi-input streaming reducers must track enough state to deterministically reproduce full-frame behavior from tile contributions.

### Reducer-specific expected input ordering

- `MeanReducer` / `GeMReducer`: `inputs == (x,)`, where `x` is `[N, C, H, W]`.
- `AttentionGeMReducer`: `inputs == (x, att_logits)`, where `att_logits` is `[N, H, W]`, `[N, 1, H, W]`, or `[N, C, H, W]`.
- `FusedAttentionGeMReducer`: `inputs == (y1, y2, y3, att_logits1, att_logits2, att_logits3)`, where all value maps are spatially aligned `[N, C, H, W]` tensors and each attention-logit map follows the `AttentionGeMReducer` logit shape contract.
- Custom reducers: explicitly document ordering (for example `(x, weights)` or `(x, guidance, confidence)`) and enforce with runtime checks.

## Package structure

- `__init__.py`
  - Public API surface for reducer classes.
  - Re-exports canonical reducer types used by the rest of the project.

- `utils.py`
  - Shared helper functions for reducer implementations.
  - `normalize_spatial_mask`: validates and canonicalizes user masks to `[N, 1, H, W]` bool format.
  - `resolve_accumulator_dtype`: enforces accumulator dtype policy (`float32`/`float64`).

- `base.py`
  - Streaming reducer infrastructure.
  - `StreamingReducerTileF`: custom autograd tile reducer op (forward sum/normalized-sum + backward expansion).
  - `StreamingReducer`: shared streaming lifecycle and orchestration-facing hooks:
    - stream state init/reset
    - tile accumulation
    - replay validation for debug/backward consistency
    - finalization

- `mean.py`
  - Mean/sum reducer implementations.
  - `Reducer`: non-streaming spatial reduction (`sum` or `mean`) with optional mask.
  - `StreamingMeanReducer`: convenience specialization of `StreamingReducer` for mean behavior.

- `gem.py`
  - `GeMReducer`: non-streaming GeM API entry point.
  - `StreamingGeMReducer`: streaming GeM execution implementation.

## `use_streaming=True` passthrough behavior

When a model (or conversion entrypoint) is configured with `use_streaming=True`, reducer handling is:

1. Your model still instantiates non-streaming `*Reducer` modules.
2. SCNN conversion walks modules and calls reducer `to_streaming()` hooks.
3. The returned `Streaming*Reducer` classes are used only in streaming execution.

This passthrough keeps your model code stable while switching execution strategy.

## Logic and design notes

1. **Separation of concerns**
   - SCNN owns tile traversal and destination placement.
   - Reducers own tile math, accumulation state, and reduction semantics.

2. **Single counting semantics**
   - In streaming mode, each output spatial position contributes once via `_stream_seen_mask`.
   - `valid_mask` is used to include only effective pixels for both value and count updates.

3. **Numerical stability policy**
   - Accumulation always uses at least `float32`.
   - Only `float32` and `float64` accumulators are allowed.

4. **Backward behavior**
   - Tile-level backward expands `[N, C, 1, 1]` gradients back to tile shape and reapplies masks.
   - Mean mode uses normalization from stream counts.

## Example: global mean pooling

A common use-case is converting spatial feature maps into one vector per channel before
classification. Mean pooling is preferred when you want each spatial location to
contribute equally and you want output shape stability (`[N, C, 1, 1]`) regardless of
input `H x W`.

```python
import torch
from lightstream.core.reducer import MeanReducer

# N=2, C=64, H=W=32
x = torch.randn(2, 64, 32, 32)

reducer = MeanReducer()
y = reducer(x)

print(y.shape)  # torch.Size([2, 64, 1, 1])
```

## Example: GeM with `r_init` and `learnable_r`

Use GeM when you want tunable pooling sharpness between average-like and max-like behavior.

```python
import torch
from lightstream.core.reducer import GeMReducer

# N=2, C=256, H=W=20
x = torch.randn(2, 256, 20, 20)

reducer = GeMReducer(
    r_init=3.0,
    learnable_r=True,
)

y = reducer(x)
print(y.shape)  # torch.Size([2, 256, 1, 1])
```

Under `use_streaming=True`, SCNN will call `reducer.to_streaming()` so tiled execution uses the corresponding `StreamingGeMReducer` implementation.

## AttentionGeM examples

These examples show generalized ordered inputs in both offline and SCNN-streaming contexts.

### Non-streaming AttentionGeM

```python
import torch
from lightstream.core.reducer import AttentionGeMReducer

x = torch.randn(2, 256, 20, 20)
attn = torch.sigmoid(torch.randn(2, 1, 20, 20))

reducer = AttentionGeMReducer(r_init=3.0, learnable_r=True)
y = reducer(x, attn)  # forward(*inputs, mask=None), ordered as (x, attn)

print(y.shape)  # torch.Size([2, 256, 1, 1])
```

### SCNN-streaming AttentionGeM

```python
from lightstream.core.reducer import AttentionGeMReducer

reducer = AttentionGeMReducer(r_init=3.0, learnable_r=True)
streaming_reducer = reducer.to_streaming()

# Runtime/tile orchestration must preserve ordered inputs: (x_tile, attn_tile)
streaming_reducer.reduce_tile(
    x_tile,
    attn_tile,
    valid_mask=tile_valid_mask,
)
```

If AttentionGeM tracks extra attention-denominator state, that state must be accumulated/finalized and replayed exactly as in full-frame mode.


## FusedAttentionGeM

`FusedAttentionGeMReducer` is a non-streaming reducer entrypoint for models that produce three value/probability maps and three attention-logit maps. Its exact input ordering is:

```python
(y1, y2, y3, att_logits1, att_logits2, att_logits3)
```

The reducer first fuses the three value maps with the constant `value_weights` buffer, then applies GeM to that fused value field. Each attention-logit map is independently converted into a full-frame, globally normalized softmax distribution (respecting any spatial mask). The three normalized attention-branch means are then fused with the constant `attention_weights` buffer. In other words, `attention_weights` combine already normalized attention distributions/contributions; they do **not** fuse raw logits before softmax.

Both `value_weights` and `attention_weights` are registered as non-trainable buffers, alongside the non-trainable GeM exponent `r`. They are included in module state and copied by `to_streaming()`, but they are not optimized during training.

SCNN conversion uses `StreamingFusedAttentionGeMReducer`, which keeps the public six-input reducer API but exposes a compact four-tensor internal payload `(fused_y, att_logits1, att_logits2, att_logits3)` for tiled accumulation and backward replay.

## Extension guide: custom reducers

If you add a new reducer family, provide both non-streaming and streaming pieces.

### 1) Implement non-streaming `*Reducer`

Create `MyReducer(nn.Module)` for regular model definitions.

Expected responsibilities:
- Validate input arity and per-input shape constraints (`NCHW` + any reducer-specific companions).
- Perform full-frame reduction semantics.
- Expose constructor args needed by users.
- Implement `to_streaming()`.

```python
class MyReducer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, *inputs, mask=None):
        ...

    def to_streaming(self):
        return StreamingMyReducer(...)
```

### 2) Implement corresponding `Streaming*Reducer`

Create `StreamingMyReducer(StreamingReducer)` and implement streaming hooks required by your math.

Typical hooks:
- `reduce_tile(...)` for tile-local contribution logic.
- `finalize_stream(...)` for final aggregation/normalization.
- Any extra state init/reset needed by your reducer semantics.
- Replay behavior that preserves ordered-input and mask semantics from non-streaming `forward(*inputs, mask=None)`.

### 3) Keep constructor/state parity

`to_streaming()` should pass all semantically relevant fields (for example exponents, eps values, dtype policy) into the streaming class so offline and streaming behavior stay aligned.

### 4) Export both classes

Re-export `MyReducer` and `StreamingMyReducer` via `lightstream.core.reducer.__init__` so conversion and imports remain consistent.

### 5) Contributor checklist

- Reuse `normalize_spatial_mask` and `resolve_accumulator_dtype` when applicable.
- Define/document input ordering for reducer arguments.
- Validate arity in non-streaming and streaming paths.
- Document denominator/counting semantics explicitly.
- Add tests for:
  - non-streaming reducer forward
  - `to_streaming()` conversion
  - streaming execution equivalence/acceptance within SCNN paths

## Migration notes for existing custom reducers

### Single-input custom reducers

- No behavior change is required if your reducer already behaves like `forward(x, mask=None)`.
- Optional cleanup: migrate signature style to `forward(*inputs, mask=None)` and unpack one positional input internally.

### Multi-input custom reducers

- Add explicit arity validation in both non-streaming and streaming entrypoints.
- Document and enforce positional input ordering.
- Implement streaming replay logic that reproduces full-frame multi-input behavior with identical input ordering and mask handling.
- Add/update tests for invalid arity, offline correctness, and streaming equivalence under replay.

## Limitations / keep in mind

- `StreamingReducerTileF` currently expects NCHW tensors and (when provided) a **2D** `valid_mask` for a tile.
- `normalize_spatial_mask` supports 2D/3D/4D masks only.
- Streaming count tracking is shared across channels (`[N, 1, 1, 1]`), so per-channel normalization behavior is not modeled.
- The deprecated compatibility path `lightstream.modules.reducer` remains available but should not be used for new code.
