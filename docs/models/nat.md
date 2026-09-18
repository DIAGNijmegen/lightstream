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

Only **PyTorch 2.13.0 with NATTEN 0.21.7** is supported. PyTorch 2.13.0 is the
newest official PyTorch build for which NATTEN 0.21.7 recommends and publishes
pre-built `libnatten` wheels in its [installation matrix](https://natten.org/install/).
Install this optional combination with:

```console
pip install 'lightstream[nat]'
```

This target uses NATTEN 0.21.7's **newer functional API**,
`natten.functional.na2d`, rather than the legacy `NeighborhoodAttention2D`
module. NATTEN is an optional dependency, so installing Lightstream without
the `nat` extra continues to support existing Lightstream models without
NATTEN.

NAT implementation code must retrieve the operation through
`lightstream.models.nat.load_na2d()`. The loader checks both installed versions
and raises `NATTENCompatibilityError` with the found and expected pair before
using an incompatible NATTEN installation. If NATTEN is absent, it instead
explains how to install the optional extra.
