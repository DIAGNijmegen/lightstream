# Global reducers

Reducers pool a spatial tensor to `[N,C,1,1]`. The equations below apply
independently to each batch item. Let \(V\) be the valid spatial positions,
\(n=|V|\), \(c\) a value channel, and \(i\) a spatial position. A spatial
`mask=` removes positions from **all** sums and normalizations. The input order
shown in the table is the positional argument order.

| Reducer | Inputs | Output for channel \(c\) |
| --- | --- | --- |
| `SumReducer` | `(x,)` | \(\sum_{i\in V}x_{ci}\) |
| `MeanReducer` | `(x,)` | \(\sum_{i\in V}x_{ci}/n\) |
| `MSEReducer` | `(x,)` | \(\sum_{i\in V}(a_{ci}-\bar a_c)^2/n\), where \(a_{ci}=x_{ci}\) or its spatial softmax |
| `GeMReducer` | `(x,)` | \(\max(\sum_{i\in V}\max(x_{ci},\epsilon)^r/n,\epsilon)^{1/r}\) |
| `NGWPReducer` | `(scores, activation_masks)` | \(\sum_{i\in V}s_{ci}m_{ci}/(\epsilon+\sum_{i\in V}m_{ci})\) |
| `SizeFocalReducer` | `(m,)` | \((1-\bar m_c)^p\log(\lambda+\bar m_c)\), where \(\bar m_c=\sum_{i\in V}m_{ci}/n\) |

`GeMReducer` can learn its exponent `r`; the attention GeM variants below keep
`r` as a fixed buffer. `NGWPReducer` defaults to `eps=1`; its activation masks
are supplied as a tensor and can be optimized separately from the optional
spatial tissue `mask=`. `SizeFocalReducer` uses fixed `p` and `lambda_`.
For an empty \(V\), sum, mean and size-focal return zero; GeM returns
\(\epsilon^{1/r}\). NGWP returns zero with its default positive eps;
eps=0 leaves an empty denominator undefined.

## Spatial consistency MSE

`MSEReducer(x)` implements the formula in the question independently for every
sample and channel. It uses the population mean over valid spatial positions:

\[
\bar a_c=\frac{1}{n}\sum_{i\in V}a_{ci},\qquad
L_c=\frac{1}{n}\sum_{i\in V}(a_{ci}-\bar a_c)^2.
\]

By default \(a_{ci}=x_{ci}\). With `use_softmax=True`, it first applies a
softmax over valid spatial positions of each channel:

\[
a_{ci}=\frac{\exp(x_{ci})}{\sum_{j\in V}\exp(x_{cj})},
\qquad \bar a_c=\frac{1}{n}.
\]

Because the probabilities sum to one, this softmax-mode MSE is at most
\((n-1)/n^2\). Its scale therefore shrinks as the number of valid instances
grows; account for this when weighting the loss. Minimizing it encourages
uniform spatial probabilities.

The output has shape `[N,C,1,1]` and stays differentiable with respect to `x`.
The softmax option can accept logits directly; inputs already normalized by a
spatial softmax should use the default mode. An empty valid set returns zero.

```python
from lightstream.core.reducer import MSEReducer

raw_consistency = MSEReducer()(instance_values)
softmax_consistency = MSEReducer(use_softmax=True)(instance_logits)
loss = softmax_consistency.mean()  # average over batch and channels if desired
```

The streaming form combines global centered moments for raw values, or a
stable softmax maximum, denominator, and squared-weight sum for softmax mode.
Its backward replay uses the resulting global mean and normalization.
For raw values the per-position slope is \(2(x_{ci}-\bar x_c)/n\). In softmax
mode it is \(2a_{ci}(a_{ci}-\sum_j a_{cj}^2)/n\); both use full-image
statistics, even when replaying one tile.

## Attention from a separate model branch

`SoftmaxAttentionReducer(values, attention_logits)` and
`AttentionGeMReducer(x, attention_logits)` accept attention logits produced
elsewhere in the model. The value map has shape `[N,C,H,W]`. Attention can be
`[N,H,W]`, `[N,1,H,W]`, or `[N,C,H,W]`; in the last case, its channels are
averaged **before** normalization to make one spatial attention field shared
by the value channels. Each reducer instance pools independently.

Write \(\ell_i\) for that shared field. Over the valid spatial domain,

\[
a_i=\frac{\exp(\ell_i)}{\sum_{j\in V}\exp(\ell_j)},\qquad
\sum_{i\in V}a_i=1.
\]

`SoftmaxAttentionReducer` leaves its instance values unchanged:

\[
y_c=\sum_{i\in V}a_i\,values_{ci}.
\]

For `AttentionGeMReducer`, write \(u_{ci}=\max(x_{ci},\epsilon)\) and
\(\alpha=\texttt{uniform_attention_eps}\). It computes

\[
b_i=(1-\alpha)a_i+\frac{\alpha}{n},\qquad
y_c=\max\left(\sum_{i\in V}b_i u_{ci}^{r},\epsilon\right)^{1/r}.
\]

The optional uniform mixture is over valid positions and defaults to zero.
It does not alter the GeM input clamp: negative values are floored before
the power, and have zero value-path gradient below that floor. A fully
masked sample returns zero for softmax attention and the GeM floor
\(\epsilon^{1/r}\) for attention GeM.

At the exact GeM output clamp boundary, full-frame and tiled summation may
round to opposite sides of `eps`. Their values stay close, while the hard
clamp can make their gradients differ discontinuously at that threshold.

Both constructors accept `stopgrad_attention=True`. This keeps the same
attention weights in the forward pass, but treats them as constants during
backpropagation. Values remain differentiable. The default is `False`, which
preserves gradients through the attention producer. For a single output
channel and an unclamped GeM result, the attention-logit derivatives are

\[
\frac{\partial y_c^{\rm softmax}}{\partial\ell_i}
=a_i(values_{ci}-y_c),\qquad
\frac{\partial y_c^{\rm GeM}}{\partial\ell_i}
=\frac{1-\alpha}{r}M_c^{1/r-1}a_i(u_{ci}^{r}-A_c),
\]

where \(A_c=\sum_j a_j u_{cj}^{r}\) and
\(M_c=\sum_j b_j u_{cj}^{r}\). With detached attention these
derivatives are zero. A shared upstream network can still receive gradients
from the value branch.

```python
from lightstream.core.reducer import AttentionGeMReducer, SoftmaxAttentionReducer

gem = AttentionGeMReducer(r_init=3.0, stopgrad_attention=True)
softmax = SoftmaxAttentionReducer(stopgrad_attention=True)

pooled_positive = gem(values, attention_logits)
pooled_raw_logits = softmax(instance_logits, attention_logits)
```

Streaming execution uses a global running softmax maximum, denominator and
weighted numerator; when the maximum changes, previous sums are rescaled.
GeM also tracks the valid uniform sum and count. SCNN's setup pass keeps both
branches connected to determine tile extents. Detachment applies only to the
actual reduction and backward replay. Replay uses the finalized global
normalization, and each spatial position contributes once even when tiles
overlap. Both attention branches must themselves support tiled replay.

## Other attention reducers

| Reducer | Inputs | Attention and pooled value |
| --- | --- | --- |
| `NormalizedSigmoidAttentionReducer` | `(values, attention_logits)` | \(q_i=\sigma(\ell_i)/\sum_{j\in V}\sigma(\ell_j)\); \(y_c=\sum_i q_i values_{ci}\). Uses the same shared-field shape rule as above. |
| `LogitAttentionPoolingReducer` | `(z,)` | For each class separately, \(a_{ci}=\operatorname{softmax}_{i\in V}(z_{ci}/\tau)\), \(y_c=\sum_i a_{ci}z_{ci}\). |
| `SigmoidAttentionPoolingReducer` | `(z,)` | For each class separately, \(a_{ci}=\operatorname{softmax}_{i\in V}(\sigma(z_{ci})/\tau)\), \(y_c=\sum_i a_{ci}z_{ci}\). |

The two single-input pooling reducers can detach their attention scores with
`stopgrad_attention`; their temperature `tau` may be fixed or learned.
All three return zero for an empty valid set.

`FusedAttentionGeMReducer(y1, y2, y3, att_logits1, att_logits2, att_logits3)`
first forms \(v=\sum_{k=1}^{3}w_k y_k\). It normalizes each attention branch
separately, then forms \(a_i=\sum_{k=1}^{3}h_k
\operatorname{softmax}_{j\in V}(\ell_{kj})_i\). After the same optional
uniform mixture \(b_i=(1-\alpha)a_i+\alpha/n\), it returns
\(\max(\sum_i b_i\max(v_{ci},\epsilon)^r,\epsilon)^{1/r}\).
Its fusion weights and exponent are fixed buffers. This reducer does not have
the new `stopgrad_attention` option. Its empty-mask output is the GeM floor.

## Spatial attention KL

`AttentionKLDivergenceReducer(student_logits, teacher_logits)` requires equal
`[N,C,H,W]` shapes and normalizes each class plane separately. With teacher
temperature \(T\),

\[
q_{ci}=\frac{\sigma(teacher_{ci}/T)}{\sum_{j\in V}\sigma(teacher_{cj}/T)},
\quad p_{ci}=\operatorname{softmax}_{i\in V}(student_{ci}),
\quad KL_c=\sum_{i\in V}q_{ci}\log\frac{q_{ci}}{p_{ci}}.
\]

The teacher is always detached. The output is a spatial **sum** per batch
and class; average across samples and classes in the caller if desired.
An empty valid set returns zero. This computes a KL objective, rather than
pooling instance values.
