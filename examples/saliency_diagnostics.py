"""Shared reporting for the opt-in StreamingCNN saliency diagnostics."""

from __future__ import annotations

import torch


def compare_saliency_candidates(
    stream_network,
    reference: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, bool]:
    """Compare diagnostic maps with a reference using the requested tolerances."""
    maps = getattr(stream_network, "saliency_diagnostic_maps", {})
    if not maps:
        print("Saliency candidates are disabled or were not produced.")
        return {}

    reference = reference.detach()
    reference_support = reference.abs() > atol
    first_failure = None
    results = {}
    for name in ("raw", "grad_lost", "ownership", "production"):
        candidate = maps.get(name)
        if candidate is None:
            print(f"\nSaliency candidate {name}: unavailable (allocation failed)")
            first_failure = first_failure or name
            results[name] = False
            continue
        candidate = candidate.to(device=reference.device, dtype=reference.dtype)
        if candidate.shape != reference.shape:
            print(f"\nSaliency candidate {name}: shape mismatch {tuple(candidate.shape)} != {tuple(reference.shape)}")
            first_failure = first_failure or name
            results[name] = False
            continue

        candidate_support = candidate.abs() > atol
        missing = reference_support & ~candidate_support
        extra = candidate_support & ~reference_support
        absolute_error = (candidate - reference).abs()
        supported_error = absolute_error[reference_support]
        tolerance = atol + rtol * reference.abs()
        mismatch = absolute_error > tolerance
        spatial_error = mismatch.any(dim=tuple(range(mismatch.ndim - 2)))
        coordinates = spatial_error.nonzero(as_tuple=False)
        error_box = None
        if coordinates.numel():
            minimum = coordinates.min(dim=0).values.tolist()
            maximum = coordinates.max(dim=0).values.tolist()
            error_box = (minimum[0], minimum[1], maximum[0] + 1, maximum[1] + 1)

        reduce_dims = tuple(range(mismatch.ndim - 2))
        mismatch_spatial_counts = mismatch.sum(dim=reduce_dims)
        row_counts = mismatch_spatial_counts.sum(dim=1).tolist()
        column_counts = mismatch_spatial_counts.sum(dim=0).tolist()
        mean_error = supported_error.mean().item() if supported_error.numel() else 0.0
        max_error = absolute_error.max().item() if absolute_error.numel() else 0.0
        failed = bool(missing.any() or extra.any() or mismatch.any())
        results[name] = not failed
        if failed and first_failure is None:
            first_failure = name
        print(
            f"\nSaliency candidate {name}:\n"
            f"  missing reference support: {missing.count_nonzero().item()}\n"
            f"  extra streamed support: {extra.count_nonzero().item()}\n"
            f"  reference-support mean absolute error: {mean_error:.6e}\n"
            f"  exact maximum absolute error: {max_error:.17e}\n"
            f"  error bounding box [top, left, bottom, right): {error_box}\n"
            f"  per-row tolerance-mismatch counts: {row_counts}\n"
            f"  per-column tolerance-mismatch counts: {column_counts}"
            f"\n  tolerance check (rtol={rtol:.3e}, atol={atol:.3e}): "
            f"{'PASS' if not failed else 'FAIL'}"
        )

    print(f"\nEarliest failing saliency transformation: {first_failure or 'none'}")
    return results
