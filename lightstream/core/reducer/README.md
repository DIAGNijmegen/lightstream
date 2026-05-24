# `lightstream.core.reducer`

This package contains the spatial reduction logic used by Lightstream in both non-streaming and streaming paths.

## Quick usage contract (important)

- Use `*Reducer` classes (for example, `MeanReducer`, `GeMReducer`) in model definitions.
- SCNN converts reducers for tiled execution via `to_streaming()`.
- `Streaming*Reducer` classes are execution implementations, **not** the primary user entrypoint.

In practice: define your model with non-streaming reducer modules, then let SCNN/runtime conversion handle the streaming class swap.

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

## Extension guide: custom reducers

If you add a new reducer family, provide both non-streaming and streaming pieces.

### 1) Implement non-streaming `*Reducer`

Create `MyReducer(nn.Module)` for regular model definitions.

Expected responsibilities:
- Validate input shape (`NCHW`).
- Perform full-frame reduction semantics.
- Expose constructor args needed by users.
- Implement `to_streaming()`.

```python
class MyReducer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x, mask=None):
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

### 3) Keep constructor/state parity

`to_streaming()` should pass all semantically relevant fields (for example exponents, eps values, dtype policy) into the streaming class so offline and streaming behavior stay aligned.

### 4) Export both classes

Re-export `MyReducer` and `StreamingMyReducer` via `lightstream.core.reducer.__init__` so conversion and imports remain consistent.

### 5) Contributor checklist

- Reuse `normalize_spatial_mask` and `resolve_accumulator_dtype` when applicable.
- Document denominator/counting semantics explicitly.
- Add tests for:
  - non-streaming reducer forward
  - `to_streaming()` conversion
  - streaming execution equivalence/acceptance within SCNN paths

## Limitations / keep in mind

- `StreamingReducerTileF` currently expects NCHW tensors and (when provided) a **2D** `valid_mask` for a tile.
- `normalize_spatial_mask` supports 2D/3D/4D masks only.
- Streaming count tracking is shared across channels (`[N, 1, 1, 1]`), so per-channel normalization behavior is not modeled.
- The deprecated compatibility path `lightstream.modules.reducer` remains available but should not be used for new code.
