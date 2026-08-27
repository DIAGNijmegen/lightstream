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
- `NormalizedSigmoidAttentionReducer`: `inputs == (values, attention_logits)`, with values `[N,C,H,W]` and attention logits `[N,H,W]`, `[N,1,H,W]`, or `[N,C,H,W]`.
- `FusedAttentionGeMReducer`: `inputs == (y1, y2, y3, att_logits1, att_logits2, att_logits3)`, where all value maps are spatially aligned `[N, C, H, W]` tensors and each attention-logit map follows the `AttentionGeMReducer` logit shape contract.
- `NGWPReducer`: `inputs == (scores, activation_masks)`, with both tensors aligned as `[N, C, H, W]`. It returns `sum(scores * activation_masks) / (eps + sum(activation_masks))` per batch/channel.
- `SizeFocalReducer`: `inputs == (m,)`, where `m` is an activation/probability tensor `[N, C, H, W]`. It returns `(1 - mean_m)^p * log(lambda_ + mean_m)` per batch/channel.
- `SigmoidAttentionPoolingReducer`: `inputs == (logits,)`, one `[N, C, H, W]` class-logit tensor. It preserves every channel and returns `[N, C, 1, 1]`; attention is the spatial `softmax(sigmoid(logits) / tau)` per class.
- `LogitAttentionPoolingReducer`: `inputs == (logits,)`, one `[N, C, H, W]` class-logit tensor used as both values and attention logits. It returns `[N, C, 1, 1]`.
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

## nGWP and size-focal reducers

Both families accept `mask=` as a tissue mask. Only tissue pixels are included in
their sums and denominators/counts. Masks follow the package's 2D/3D/4D spatial
mask contract. By default they must already align with the reducer output; set
`mask_resize=True` to resize a reduced-resolution tissue mask using nearest-neighbor
interpolation. Empty tissue masks return zero for every affected output channel.
Mask resizing uses one global nearest-neighbor coordinate system for each reducer
output domain. Output-row chunking bounds temporary indexing memory only and does
not restart coordinates at chunk boundaries. When a streamed model has multiple
reducer heads, each head receives a separately resized mask matching its own spatial
resolution.

`StreamingNGWPReducer` retains separate weighted-score and activation-mask sums,
then divides only during `finalize_from_state`. `StreamingSizeFocalReducer` retains
only activation sums and valid-pixel counts, then applies its nonlinear expression
only during finalization. Their backward replay uses those finalized global values,
so tiled gradients equal the corresponding full-frame gradients.

These reducers are independent terminal outputs in a streamed model; do not compose
them in the streaming graph. Once `StreamingCNN.forward()` has completed, combine
their finalized `[N, C, 1, 1]` outputs in the caller/training module:

```python
final_prediction = ngwp_prediction + size_focal_prediction
```

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

## Logit-attention pooling

`LogitAttentionPoolingReducer` accepts exactly one class-logit tensor
`z` of shape `[N, C, H, W]`. The same tensor supplies the pooled values and
the attention scores. For every batch and class independently it computes

```text
q_i = z_i / tau
a_i = exp(q_i) / sum_j exp(q_j)
y = sum_i a_i z_i                         # [N, C, 1, 1]
```

The positive temperature is parameterized as
`tau = tau_min + softplus(raw_tau)`. It may be fixed or learned. Its parameter
storage retains its own dtype and device across `to_streaming()` and
`to_reducer()` conversions; `accumulator_dtype` controls only reduction math.
Both `tau_init` and `tau_min` are validated, and `tau_init` must be finite and
strictly greater than a finite, non-negative `tau_min`.

An optional spatial `mask` excludes positions before softmax normalization.
Masks use the package-wide mask shapes and resizing rules (`mask_resize` and
`mask_resize_mode`). A batch item with no valid position returns zero for every
class. The output always has the input dtype, even when accumulation uses
float32 or float64.

With `stopgrad_attention=True`, the attention branch uses
`q = z.detach() / tau`: logits consequently receive only the direct value
gradient `a_i`, while a learned temperature remains trainable. With the option
disabled, logits receive the complete value-plus-softmax derivative. Streaming
backward replay uses the finalized global softmax maximum, denominator, and
mean, rather than normalizing within each tile. A global weighted-square moment
provides the learned-temperature derivative once per backward replay, so it is
independent of tile count and overlap.

This differs from `SigmoidAttentionPoolingReducer` only in the attention source:
sigmoid attention uses `softmax(sigmoid(z) / tau)`, bounding attention scores,
whereas logit attention uses `softmax(z / tau)` directly. Both pool the original
logits, normalize spatially per batch/class, share masking and temperature
configuration, and expose streaming classes only as execution implementations.

## AttentionGeM

## Normalized sigmoid attention

`NormalizedSigmoidAttentionReducer` preserves `values` exactly and uses the ratio

```text
output_c = sum_i(sigmoid(attention_logits_i) * values_i,c)
           / sum_i(sigmoid(attention_logits_i))
```

over valid spatial positions. It does not apply a sigmoid, clamp, power, or any
other transformation to instance values. One-channel attention broadcasts across
value channels. As in `AttentionGeMReducer`, `[N,C,H,W]` attention logits are
averaged across channels to a single attention field before sigmoid and spatial
normalization. Masks exclude pixels from both sums, and fully masked samples yield
zero. `StreamingNormalizedSigmoidAttentionReducer` accumulates the same numerator
and denominator across tiles and uses global statistics during backward replay.

`AttentionGeMReducer` reduces a value tensor `x` with attention logits over the same spatial domain. Its exact input ordering is:

```python
(x, att_logits)
```

`att_logits` may be `[N, H, W]`, `[N, 1, H, W]`, or `[N, C, H, W]`. Channel-wise logits are averaged to one spatial logit field before normalization. The reducer converts logits to a globally normalized softmax over valid spatial positions, applies that attention to `x.clamp_min(eps) ** r`, and finally applies the GeM root.

### Uniform attention mixing

`uniform_attention_eps` controls an optional blend between learned attention and a uniform distribution over valid spatial locations. It is validated as a finite value in `[0, 1]` and defaults to `0.0`, preserving pure attention behavior.

For each valid spatial position `i`, let `a_i` be the softmax-normalized attention weight after masking. The mixed weight is:

```text
a_prime_i = (1 - eps) * a_i + eps / N_valid
```

where `eps` is `uniform_attention_eps`. Invalid positions receive zero contribution. Masks define `N_valid`: with `mask`, `N_valid` is the count of `True` positions in the reducer's spatial domain; without a mask, `N_valid = H * W`. Thus `uniform_attention_eps=1.0` is equivalent to masked/unmasked uniform GeM over the valid positions, while intermediate values keep a learned-attention term.

The same semantics are used by `StreamingAttentionGeMReducer`: streaming accumulation tracks both the attention numerator/denominator and the valid uniform sum/count so tiled forward and backward replay match full-frame reduction.

### Non-streaming AttentionGeM

```python
import torch
from lightstream.core.reducer import AttentionGeMReducer

x = torch.randn(2, 256, 20, 20)
att_logits = torch.randn(2, 1, 20, 20)

reducer = AttentionGeMReducer(r_init=3.0, uniform_attention_eps=0.05)
y = reducer(x, att_logits)  # forward(*inputs, mask=None), ordered as (x, att_logits)

print(y.shape)  # torch.Size([2, 256, 1, 1])
```

### SCNN-streaming AttentionGeM

```python
from lightstream.core.reducer import AttentionGeMReducer

reducer = AttentionGeMReducer(r_init=3.0, uniform_attention_eps=0.05)
streaming_reducer = reducer.to_streaming()

# Runtime/tile orchestration must preserve ordered inputs: (x_tile, att_logits_tile).
# SCNN calls accumulate_stream_tile/finalize internally; users typically do not.
```


## FusedAttentionGeM

`FusedAttentionGeMReducer` is a non-streaming reducer entrypoint for models that produce three value/probability maps and three attention-logit maps. Its exact input ordering is:

```python
(y1, y2, y3, att_logits1, att_logits2, att_logits3)
```

The reducer first fuses the three value maps with the constant `value_weights` buffer, then applies GeM to that fused value field. Each attention-logit map is independently converted into a full-frame, globally normalized softmax distribution (respecting any spatial mask). The three normalized attention branches are then fused with the constant `attention_weights` buffer. In other words, `attention_weights` combine already normalized attention distributions/contributions; they do **not** fuse raw logits before softmax.

`uniform_attention_eps` has the same valid-position definition as regular `AttentionGeMReducer`, but the uniform mix is applied **after** branch fusion, not per branch. Equivalently, form the fused attention weight first:

```text
a_i = sum_j attention_weights[j] * softmax(att_logits_j)_i
a_prime_i = (1 - eps) * a_i + eps / N_valid
```

Again, masks define `N_valid`; without a mask, `N_valid = H * W`. This post-fusion rule means the uniform term is added once to the fused attention distribution rather than separately to each attention branch.

Both `value_weights` and `attention_weights` are registered as non-trainable buffers, alongside the non-trainable GeM exponent `r`. They are included in module state and copied by `to_streaming()`, but they are not optimized during training.

SCNN conversion uses `StreamingFusedAttentionGeMReducer`, which keeps the public six-input reducer API but exposes a compact two-tensor internal payload `(fused_y, att_logits_stacked)`, where the stacked logits use shape `[N, 3, H, W]` for tiled accumulation and backward replay. The streaming implementation tracks per-branch softmax state plus one fused valid uniform sum/count, preserving the post-branch-fusion uniform-mix semantics in forward and backward replay.

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

## Sigmoid attention pooling

`SigmoidAttentionPoolingReducer` accepts exactly one class-logit tensor in NCHW
layout. For every batch item and class independently it computes
`softmax(sigmoid(logits) / tau)` across valid spatial positions and uses those
weights to sum the original, raw logits. No class channels are combined. The
result is always `[N, C, 1, 1]`. `tau` may be fixed or learned through a positive
softplus parameterization. With `stopgrad_attention=True`, gradients through the
sigmoid scores are stopped while the value path and a learned temperature remain
differentiable. Its `StreamingSigmoidAttentionPoolingReducer` counterpart is an
execution implementation; model definitions should instantiate the offline class
and let `to_streaming()` perform conversion.
