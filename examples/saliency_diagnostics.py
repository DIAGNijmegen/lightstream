"""Shared reporting for the opt-in StreamingCNN saliency diagnostics."""

from __future__ import annotations

import torch


_PREVIEW_LIMIT = 20


def _format_sides(sides: dict) -> str:
    return (
        "Sides("
        + ", ".join(
            f"{side}={bool(sides.get(side, False))}"
            for side in ("top", "left", "bottom", "right")
        )
        + ")"
    )


def _report_missing_additive_writes(
    stream_network, raw, candidate, name, reference, tolerance, *, verbose
) -> None:
    """Report non-hole errors associated with fewer additive tile writes."""
    count_maps = getattr(stream_network, "saliency_diagnostic_write_count_maps", None)
    if count_maps is None:
        count_maps = getattr(stream_network, "saliency_diagnostic_count_maps", {})
    raw_counts = count_maps.get("raw") if count_maps else None
    candidate_counts = count_maps.get(name) if count_maps else None
    if raw_counts is None or candidate_counts is None:
        return

    raw_counts = raw_counts.to(reference.device)
    candidate_counts = candidate_counts.to(reference.device)
    mask = (
        raw.ne(0)
        & candidate.ne(0)
        & raw_counts.gt(candidate_counts)
        & (candidate - reference).abs().gt(tolerance)
    )
    spatial_mask = mask.any(dim=tuple(range(mask.ndim - 2)))
    coordinates = spatial_mask.nonzero(as_tuple=False)
    total = coordinates.shape[0]
    displayed = coordinates if verbose else coordinates[:_PREVIEW_LIMIT]
    preview = [tuple(point) for point in displayed.tolist()]
    omitted = total - len(preview)
    suffix = f" ... {omitted} additional coordinates omitted" if omitted else ""
    bounding_box = None
    if total:
        minimum = coordinates.min(dim=0).values.tolist()
        maximum = coordinates.max(dim=0).values.tolist()
        bounding_box = (minimum[0], minimum[1], maximum[0] + 1, maximum[1] + 1)
    print(
        f"  nonzero coordinates with fewer writes than raw and tolerance-significant "
        f"error: total={total}, coordinates={preview}{suffix}, "
        f"bounding box [top, left, bottom, right)={bounding_box}"
    )

    print("  missing-additive-write summary by tile boundary and Sides:")
    found = False
    for record in getattr(stream_network, "saliency_diagnostic_records", []):
        boundaries = record.get("candidate_destination_slices", {})
        boundary = boundaries.get(name)
        if not boundary:
            continue
        top, left, bottom, right = boundary
        count = int(spatial_mask[top:bottom, left:right].count_nonzero().item())
        if count:
            found = True
            print(
                f"    boundary={(top, left, bottom, right)}, "
                f"{_format_sides(record.get('sides', {}))}: {count} coordinates"
            )
    if not found:
        print("    none")


def compare_saliency_candidates(
    stream_network,
    reference: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    verbose: bool = False,
    diagnose_assembly: bool = False,
) -> dict[str, bool]:
    """Report saliency stages and assert parity for the supported candidates.

    ``raw`` and ``production`` are regression candidates.  The intermediate
    ``grad_lost`` and ``ownership`` maps exist to characterize the historical
    transformations and are deliberately not part of the regression result.
    Set ``diagnose_assembly`` to report the counterfactual assembly stages and
    ``verbose`` to print their complete per-row and per-column mismatch arrays.
    The default report is limited to the production regression result.
    """
    maps = getattr(stream_network, "saliency_diagnostic_maps", {})
    if not maps:
        print("Saliency candidates are disabled or were not produced.")
        return {}

    reference = reference.detach()
    reference_support = reference.abs() > atol
    if not diagnose_assembly:
        production = maps.get("production")
        if production is None or production.shape != reference.shape:
            raise AssertionError("production saliency candidate is unavailable or has the wrong shape")
        production = production.to(device=reference.device, dtype=reference.dtype)
        production_support = production.abs() > atol
        missing = reference_support & ~production_support
        extra = production_support & ~reference_support
        error = (production - reference).abs()
        supported_error = error[reference_support]
        passed = not bool(
            missing.any() or extra.any() or (error > atol + rtol * reference.abs()).any()
        )
        print(
            "Saliency production comparison:\n"
            f"  reference support: {reference_support.count_nonzero().item()}\n"
            f"  production support: {production_support.count_nonzero().item()}\n"
            "  production reference-support mean absolute error: "
            f"{supported_error.mean().item() if supported_error.numel() else 0.0:.6e}\n"
            f"  production maximum absolute error: {error.max().item() if error.numel() else 0.0:.17e}\n"
            f"  production tolerance check (rtol={rtol:.3e}, atol={atol:.3e}): "
            f"{'PASS' if passed else 'FAIL'}"
        )
        raw = maps.get("raw")
        raw_parity = True
        if raw is not None and raw.shape == production.shape:
            raw = raw.to(device=reference.device, dtype=reference.dtype)
            raw_parity = bool(torch.all(
                (raw - production).abs() <= atol + rtol * production.abs()
            ))
            print(f"Raw/production parity check: {'PASS' if raw_parity else 'FAIL'}")
        if not passed:
            raise AssertionError("production saliency candidate failed reference tolerance")
        if not raw_parity:
            raise AssertionError("raw and production saliency candidates differ")
        return {"production": passed, **({"raw": raw_parity} if raw is not None else {})}

    earliest_destructive = None
    results = {}
    for name in ("raw", "grad_lost", "ownership", "production"):
        candidate = maps.get(name)
        if candidate is None:
            print(f"\nSaliency candidate {name}: unavailable (allocation failed)")
            results[name] = False
            continue
        candidate = candidate.to(device=reference.device, dtype=reference.dtype)
        if candidate.shape != reference.shape:
            print(
                f"\nSaliency candidate {name}: shape mismatch {tuple(candidate.shape)} != {tuple(reference.shape)}"
            )
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
        row_counts = mismatch_spatial_counts.sum(dim=1)
        column_counts = mismatch_spatial_counts.sum(dim=0)
        nonzero_row_indices = row_counts.nonzero(as_tuple=False).flatten()
        nonzero_column_indices = column_counts.nonzero(as_tuple=False).flatten()
        nonzero_rows = [
            (int(index), int(row_counts[index]))
            for index in nonzero_row_indices[:_PREVIEW_LIMIT]
        ]
        nonzero_columns = [
            (int(index), int(column_counts[index]))
            for index in nonzero_column_indices[:_PREVIEW_LIMIT]
        ]
        mean_error = supported_error.mean().item() if supported_error.numel() else 0.0
        max_error = absolute_error.max().item() if absolute_error.numel() else 0.0
        failed = bool(missing.any() or extra.any() or mismatch.any())
        results[name] = not failed
        is_characterization = name in ("grad_lost", "ownership")
        if failed and is_characterization and earliest_destructive is None:
            earliest_destructive = name
        tolerance_result = (
            "EXPECTED DIVERGENCE"
            if failed and is_characterization
            else "FAIL" if failed else "PASS"
        )
        print(
            f"\nSaliency candidate {name}:\n"
            f"  missing reference support: {missing.count_nonzero().item()}\n"
            f"  extra streamed support: {extra.count_nonzero().item()}\n"
            f"  reference-support mean absolute error: {mean_error:.6e}\n"
            f"  exact maximum absolute error: {max_error:.17e}"
        )
        if mismatch.any():
            print(
                f"  total mismatching rows: {nonzero_row_indices.numel()}\n"
                f"  total mismatching columns: {nonzero_column_indices.numel()}\n"
                f"  maximum mismatches in any row: {row_counts.max().item()}\n"
                f"  maximum mismatches in any column: {column_counts.max().item()}\n"
                "  first nonzero row mismatch counts (index, count): "
                f"{nonzero_rows}\n"
                "  first nonzero column mismatch counts (index, count): "
                f"{nonzero_columns}\n"
                f"  error bounding box [top, left, bottom, right): {error_box}"
            )
        else:
            print(
                "  all tolerance-mismatch counts are zero "
                "(per-row and per-column: all zero)"
            )
        if verbose:
            print(
                f"  per-row tolerance-mismatch counts: {row_counts.tolist()}\n"
                f"  per-column tolerance-mismatch counts: {column_counts.tolist()}"
            )
        print(
            f"  tolerance check (rtol={rtol:.3e}, atol={atol:.3e}): "
            f"{tolerance_result}"
        )
        if name in ("grad_lost", "ownership"):
            raw = maps.get("raw")
            if raw is not None and raw.shape == reference.shape:
                _report_missing_additive_writes(
                    stream_network,
                    raw.to(device=reference.device, dtype=reference.dtype),
                    candidate,
                    name,
                    reference,
                    tolerance,
                    verbose=verbose,
                )

    print(
        "\nEarliest destructive saliency transformation: "
        f"{earliest_destructive or 'none'}"
    )

    parity_failures = [
        name for name in ("raw", "production") if not results.get(name, False)
    ]
    if not parity_failures:
        raw = maps["raw"].to(device=reference.device, dtype=reference.dtype)
        production = maps["production"].to(
            device=reference.device, dtype=reference.dtype
        )
        try:
            torch.testing.assert_close(
                raw,
                production,
                rtol=rtol,
                atol=atol,
                msg="raw and production saliency candidates differ",
            )
            print("Raw/production parity check: PASS")
        except AssertionError:
            print("Raw/production parity check: FAIL")
            raise
    else:
        raise AssertionError(
            "Saliency parity candidate(s) failed reference tolerance: "
            + ", ".join(parity_failures)
        )
    return results
