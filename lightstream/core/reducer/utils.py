"""Shared reducer helper utilities for mask and dtype handling."""

import torch


def normalize_spatial_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Normalize user-provided mask to ``[N, 1, H, W]`` boolean format.

    Parameters
    ----------
    mask : torch.Tensor
        Input mask in 2D (H,W), 3D (N,H,W), or 4D (N,1|C,H,W) layout.
    x : torch.Tensor
        Reference tensor used for batch/spatial shape validation.

    Returns
    -------
    torch.Tensor
        Normalized boolean mask with shape ``[N, 1, H, W]``.
    """
    if mask.ndim == 2:
        if mask.shape != x.shape[-2:]:
            raise ValueError(f"mask shape {tuple(mask.shape)} must match input spatial shape {tuple(x.shape[-2:])}")
        return mask[None, None].to(device=x.device, dtype=torch.bool)

    if mask.ndim == 3:
        if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"3D mask shape {tuple(mask.shape)} must be [N,H,W] with N={x.shape[0]}, H/W={tuple(x.shape[-2:])}"
            )
        return mask[:, None].to(device=x.device, dtype=torch.bool)

    if mask.ndim == 4:
        if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"4D mask shape {tuple(mask.shape)} must be [N,1,H,W] with N={x.shape[0]}, H/W={tuple(x.shape[-2:])}"
            )
        if mask.shape[1] not in (1, x.shape[1]):
            raise ValueError(
                f"4D mask channel dim must be 1 or C={x.shape[1]}, got {mask.shape[1]}"
            )
        mask_bool = mask.to(device=x.device, dtype=torch.bool)
        if mask_bool.shape[1] == x.shape[1]:
            mask_bool = torch.any(mask_bool, dim=1, keepdim=True)
        return mask_bool

    raise ValueError(f"mask must be 2D/3D/4D spatial mask, got shape={tuple(mask.shape)}")


def resolve_accumulator_dtype(accumulator_dtype: torch.dtype | None, reference_dtype: torch.dtype) -> torch.dtype:
    """Resolve an accumulator dtype with minimum precision constraints.

    Parameters
    ----------
    accumulator_dtype : torch.dtype | None
        Requested accumulator dtype.
    reference_dtype : torch.dtype
        Input tensor dtype used as fallback.

    Returns
    -------
    torch.dtype
        Resolved dtype, restricted to ``torch.float32`` or ``torch.float64``.
    """
    if accumulator_dtype is None:
        resolved = reference_dtype if reference_dtype in (torch.float32, torch.float64) else torch.float32
    else:
        resolved = accumulator_dtype
    if resolved not in (torch.float32, torch.float64):
        raise ValueError(
            f"Unsupported accumulator_dtype '{resolved}'. Use torch.float32 or torch.float64."
        )
    return resolved
