"""Shared reducer helper utilities for mask and dtype handling."""

import torch
import torch.nn.functional as F


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


def prepare_spatial_mask(
    mask: torch.Tensor,
    x: torch.Tensor,
    *,
    mask_resize: bool = False,
    mask_resize_mode: str = "nearest",
) -> torch.Tensor:
    """Prepare a spatial mask for reduction against ``x``.

    Masks with spatial dimensions matching ``x`` are normalized directly. When
    spatial dimensions differ, optionally resize with nearest-neighbor
    interpolation before normalizing to ``[N, 1, H, W]`` boolean format on
    ``x.device``.
    """
    if mask_resize_mode != "nearest":
        raise ValueError(f"Unsupported mask_resize_mode '{mask_resize_mode}'. Only 'nearest' is supported.")

    if mask.ndim not in (2, 3, 4):
        raise ValueError(f"mask must be 2D/3D/4D spatial mask, got shape={tuple(mask.shape)}")

    target_spatial = tuple(x.shape[-2:])
    mask_spatial = tuple(mask.shape[-2:])
    if mask_spatial == target_spatial:
        return normalize_spatial_mask(mask, x)

    if not mask_resize:
        raise ValueError(
            f"mask spatial shape {mask_spatial} must match input spatial shape {target_spatial}; "
            "set mask_resize=True to resize masks with nearest-neighbor interpolation"
        )

    if mask.ndim == 2:
        mask_nchw = mask[None, None]
    elif mask.ndim == 3:
        if mask.shape[0] != x.shape[0]:
            raise ValueError(
                f"3D mask shape {tuple(mask.shape)} must be [N,H,W] with N={x.shape[0]} before resizing"
            )
        mask_nchw = mask[:, None]
    else:
        if mask.shape[0] != x.shape[0]:
            raise ValueError(
                f"4D mask shape {tuple(mask.shape)} must have N={x.shape[0]} before resizing"
            )
        if mask.shape[1] not in (1, x.shape[1]):
            raise ValueError(
                f"4D mask channel dim must be 1 or C={x.shape[1]}, got {mask.shape[1]}"
            )
        mask_nchw = mask

    resized = F.interpolate(
        mask_nchw.to(device=x.device, dtype=torch.float32),
        size=target_spatial,
        mode=mask_resize_mode,
    ).to(dtype=torch.bool)
    return normalize_spatial_mask(resized, x)


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
