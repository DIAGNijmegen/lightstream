# Neighborhood Attention Transformer (NAT)

## Initial streaming target

The initial Lightstream target is the **standalone NAT backbone** with this
deliberately narrow contract:

- batch size **1**;
- an odd neighborhood-attention kernel, initially **kernel size 7**;
- dilation fixed to **1**;
- evaluation mode;
- dropout, attention dropout, and stochastic depth disabled; and
- four backbone feature maps in **NCHW** layout.

Equivalence work must cover the forward result, the input gradient, and every
parameter gradient between conventional and streaming execution.

The initial target explicitly excludes **DiNAT**, **DiNAT-s**, MMDetection,
detection heads, training-mode dropout, and stochastic depth. These exclusions
are not claims that those configurations can never be supported; they are
outside the first validation boundary.

## Supported dependency and API

Only **PyTorch 2.13.0 with the NATTEN `v0.17.5-blackwell` build** is supported.
That NATTEN branch identifies itself as the ordinary Python version `0.17.5`,
so a requirement such as `natten==0.17.5-blackwell` is not valid package
metadata and `natten==0.17.5` could silently select the non-Blackwell release.
The `nat` extra therefore uses the production branch's immutable source commit
`0a6a3df544fe478e97bdd77e92fd3ec14cf43cff` rather than a version specifier.

Install the supported combination (and build NATTEN for the CUDA architecture
visible on the build host) with:

```console
pip install 'lightstream[nat]'
```

For a controlled Blackwell build where CUDA is not visible during installation,
install PyTorch first and force the desired architecture while installing the
same immutable source revision:

```console
pip install 'torch==2.13.0'
NATTEN_WITH_CUDA=1 NATTEN_CUDA_ARCH=10.0 pip install \
  'natten @ git+https://github.com/SHI-Labs/NATTEN.git@0a6a3df544fe478e97bdd77e92fd3ec14cf43cff'
pip install --no-deps lightstream
```

Change `NATTEN_CUDA_ARCH` only when the production GPU has a different compute
capability. NATTEN is optional, so installing Lightstream without the `nat`
extra continues to support models that do not use neighborhood attention.
At runtime Lightstream checks for package version `0.17.5`, and its NAT layer
uses the `NeighborhoodAttention2D` constructor with `rel_pos_bias=True`
explicitly. The immutable dependency pin is what distinguishes
the supported Blackwell source from other builds that report the same version.

## NCHW NAT layers and checkpoints

`lightstream.models.nat.NCHWNATLayer` is the streaming-friendly NAT layer. It
uses NCHW tensors, channel-wise layer normalization, explicit streaming-aware
residual additions, and 1x1 convolutions for the two MLP projections. Construct
it with `channels`, `num_heads`, and the usual NAT kernel, dilation, QKV, scale,
and dropout options, or pass an existing NATTEN attention module through the
`attention` argument when converting an existing model.

Original NAT checkpoints retain their existing key names. `PointwiseConvMlp`
accepts the original two-dimensional `mlp.fc1.weight` and `mlp.fc2.weight`
tensors while loading and reshapes them automatically. For conversion outside
a model load, use `convert_nhwc_nat_state_dict`; use
`convert_nchw_nat_state_dict` for the inverse operation when an NHWC NAT model
is the destination. The lower-level `linear_to_pointwise_conv` and
`pointwise_conv_to_linear` helpers convert modules while preserving parameter
values, bias presence, dtype, device, training mode, and `requires_grad` flags.
