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
from lightstream.core.reducer import Reducer

# N=2, C=64, H=W=32
x = torch.randn(2, 64, 32, 32)

reducer = Reducer(mode="mean")
y = reducer(x)

print(y.shape)  # torch.Size([2, 64, 1, 1])
```

Why this works well in Lightstream:
- It matches the expected contract of downstream linear/classification heads.
- It is robust to varying spatial sizes because reduction is normalized by pixel count.
- The same semantic mode (`mean`) maps naturally to `StreamingMeanReducer` in tiled
  execution, so offline and streaming behaviors stay aligned.

## How to add a new reducer module

Use this pattern for any new reducer variant (for example `max.py` or a true GeM implementation):

1. Add a new module in this package (e.g. `lightstream/core/reducer/myreducer.py`).
2. If the reducer is streaming-compatible, subclass `StreamingReducer` and override only reducer-specific math:
   - `reduce_tile(...)` for tile-local transform + reduction semantics.
   - `finalize_stream(...)` if final normalization/transformation differs from base behavior.
3. For non-streaming behavior, create a dedicated `nn.Module` (similar to `Reducer` in `mean.py`).
4. Reuse helpers from `utils.py` for mask and dtype handling.
5. Export the new public class in `lightstream/core/reducer/__init__.py`.
6. Keep backward compatibility in `lightstream/modules/reducer.py` if the class should be available from legacy import paths.

## Limitations / keep in mind

- `StreamingReducerTileF` currently expects NCHW tensors and (when provided) a **2D** `valid_mask` for a tile.
- `normalize_spatial_mask` supports 2D/3D/4D masks only.
- Streaming count tracking is shared across channels (`[N, 1, 1, 1]`), so per-channel normalization behavior is not modeled.
- `StreamingGeMReducer` is currently a placeholder alias over mean-style accumulation; it is **not** a full GeM power-mean implementation yet.
- The deprecated compatibility path `lightstream.modules.reducer` remains available but should not be used for new code.
