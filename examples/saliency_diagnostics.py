"""Shared reporting for the opt-in StreamingCNN saliency diagnostics."""

from __future__ import annotations

import torch


def compare_saliency_candidates(stream_network, reference: torch.Tensor) -> None:
    """Compare every allocated diagnostic map with a reference input gradient."""
    maps = getattr(stream_network, "saliency_diagnostic_maps", {})
    if not maps:
        print("Saliency candidates are disabled or were not produced.")
        return

    reference = reference.detach()
    reference_support = reference.ne(0)
    first_failure = None
    for name in ("raw", "grad_lost", "ownership", "production"):
        candidate = maps.get(name)
        if candidate is None:
            print(f"\nSaliency candidate {name}: unavailable (allocation failed)")
            continue
        candidate = candidate.to(device=reference.device, dtype=reference.dtype)
        if candidate.shape != reference.shape:
            print(f"\nSaliency candidate {name}: shape mismatch {tuple(candidate.shape)} != {tuple(reference.shape)}")
            first_failure = first_failure or name
            continue

        candidate_support = candidate.ne(0)
        missing = reference_support & ~candidate_support
        extra = candidate_support & ~reference_support
        absolute_error = (candidate - reference).abs()
        supported_error = absolute_error[reference_support]
        error_support = absolute_error.ne(0)
        spatial_error = error_support.any(dim=tuple(range(error_support.ndim - 2)))
        coordinates = spatial_error.nonzero(as_tuple=False)
        error_box = None
        if coordinates.numel():
            minimum = coordinates.min(dim=0).values.tolist()
            maximum = coordinates.max(dim=0).values.tolist()
            error_box = (minimum[0], minimum[1], maximum[0] + 1, maximum[1] + 1)

        reduce_dims = tuple(range(missing.ndim - 2))
        missing_spatial_counts = missing.sum(dim=reduce_dims)
        row_counts = missing_spatial_counts.sum(dim=1).tolist()
        column_counts = missing_spatial_counts.sum(dim=0).tolist()
        mean_error = supported_error.mean().item() if supported_error.numel() else 0.0
        max_error = supported_error.max().item() if supported_error.numel() else 0.0
        failed = bool(missing.any() or extra.any() or error_support.any())
        if failed and first_failure is None:
            first_failure = name
        print(
            f"\nSaliency candidate {name}:\n"
            f"  missing reference support: {missing.count_nonzero().item()}\n"
            f"  extra streamed support: {extra.count_nonzero().item()}\n"
            f"  reference-support absolute error: mean={mean_error:.6e}, max={max_error:.6e}\n"
            f"  error bounding box [top, left, bottom, right): {error_box}\n"
            f"  per-row missing-support counts: {row_counts}\n"
            f"  per-column missing-support counts: {column_counts}"
        )

    print(f"\nEarliest failing saliency transformation: {first_failure or 'none'}")
