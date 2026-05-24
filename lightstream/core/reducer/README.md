# `lightstream.core.reducer`

This package contains the spatial reduction logic used by Lightstream in both non-streaming and streaming paths.

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
  - `StreamingGeMReducer` API entry point.
  - Currently implemented as a compatibility alias over mean-style streaming behavior.

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
from lightstream.core.reducer import MeanReducer, SumReducer

# N=2, C=64, H=W=32
x = torch.randn(2, 64, 32, 32)

reducer = MeanReducer()
y = reducer(x)

print(y.shape)  # torch.Size([2, 64, 1, 1])
```

Why this works well in Lightstream:
- It matches the expected contract of downstream linear/classification heads.
- It is robust to varying spatial sizes because reduction is normalized by pixel count.
- The same semantic mode (`mean`) maps naturally to `StreamingMeanReducer` in tiled
  execution, so offline and streaming behaviors stay aligned.

## Tutorial: create and extend a reducer (mean example)

This tutorial shows the intended extension path using **mean** as the reference.

### 1) Non-streaming reducer: define plain reduction behavior

Create a module like `lightstream/core/reducer/mean.py` with an `nn.Module` that
accepts `x: [N, C, H, W]` and returns `[N, C, 1, 1]`.

```python
class MeanReducer(nn.Module):
    def __init__(self, mode: str = "mean", accumulator_dtype=None):
        ...

    def forward(self, x, mask=None):
        # validate x
        # optional: normalize/validate mask
        # mode == "sum": sum over H/W
        # mode == "mean": sum / count
        return y  # [N, C, 1, 1]
```

Why: this is the canonical non-streaming API used by regular model execution.

### 2) Streaming reducer: reuse shared lifecycle, customize math only if needed

For mean semantics, use the provided specialization:

```python
class StreamingMeanReducer(StreamingReducer):
    def __init__(self, accumulator_dtype=None):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
```

For a new reducer type, subclass `StreamingReducer` and override:
- `reduce_tile(...)` for tile-local reduction math
- `finalize_stream(...)` if final stream aggregation differs from sum/mean behavior

Why: SCNN handles tile orchestration; reducer classes should focus on reducer math and
state transitions.

### 3) Export it through package API

Add your class in `lightstream/core/reducer/__init__.py`:

```python
from .myreducer import StreamingMyReducer

__all__ = [
    ...
    "StreamingMyReducer",
]
```

Why: users and internal modules should import only from `lightstream.core.reducer`.

### 4) Keep behavior aligned between offline and streaming paths

For mean, alignment means:
- non-streaming: spatial mean over the full feature map
- streaming: per-tile contributions + pixel-count normalization at `finalize_stream()`

Use the same mode names/semantics (`"mean"`, `"sum"`) so conversion logic remains
predictable.

### 5) Minimal usage snippets

**Offline mean reduction**

```python
from lightstream.core.reducer import MeanReducer, SumReducer

reducer = MeanReducer()
out = reducer(x)  # [N, C, 1, 1]
```

**Streaming mean reduction class selection**

```python
from lightstream.core.reducer import StreamingMeanReducer

streaming_reducer = StreamingMeanReducer()
# SCNN/constructor orchestration will call stream lifecycle methods.
```

### 6) Checklist for contributors

- Validate tensor shapes early (`NCHW` expected).
- Reuse `normalize_spatial_mask` and `resolve_accumulator_dtype` from `utils.py`.
- Ensure counting semantics are explicit (what contributes to denominator?).
- Document differences between non-streaming and streaming behavior.
- Add/adjust tests for both direct reducer calls and SCNN-integrated execution paths.

## Limitations / keep in mind

- `StreamingReducerTileF` currently expects NCHW tensors and (when provided) a **2D** `valid_mask` for a tile.
- `normalize_spatial_mask` supports 2D/3D/4D masks only.
- Streaming count tracking is shared across channels (`[N, 1, 1, 1]`), so per-channel normalization behavior is not modeled.
- `StreamingGeMReducer` is currently a placeholder alias over mean-style accumulation; it is **not** a full GeM power-mean implementation yet.
- The deprecated compatibility path `lightstream.modules.reducer` remains available but should not be used for new code.
