from types import SimpleNamespace

import pytest
import torch

from examples.saliency_diagnostics import compare_saliency_candidates


def test_candidate_summary_uses_gradient_tolerance_for_errors(capsys):
    reference = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=torch.float64
    )
    raw = reference.clone()
    raw[0, 0, 0, 0] += 1e-17
    grad_lost = reference.clone()
    grad_lost[0, 0, 1, 2] += 2e-6
    stream_network = SimpleNamespace(
        saliency_diagnostic_maps={
            "raw": raw,
            "grad_lost": grad_lost,
            "ownership": grad_lost,
            "production": raw.clone(),
        }
    )

    results = compare_saliency_candidates(
        stream_network, reference, rtol=0.0, atol=1e-6
    )

    assert results == {
        "raw": True,
        "grad_lost": False,
        "ownership": False,
        "production": True,
    }
    report = capsys.readouterr().out
    assert "error bounding box [top, left, bottom, right): (1, 2, 2, 3)" in report
    assert "total mismatching rows: 1" in report
    assert "total mismatching columns: 1" in report
    assert "maximum mismatches in any row: 1" in report
    assert "maximum mismatches in any column: 1" in report
    assert "first nonzero row mismatch counts (index, count): [(1, 1)]" in report
    assert "first nonzero column mismatch counts (index, count): [(2, 1)]" in report
    assert "per-row tolerance-mismatch counts" not in report
    assert "per-column tolerance-mismatch counts" not in report
    assert (
        "tolerance check (rtol=0.000e+00, atol=1.000e-06): EXPECTED DIVERGENCE"
        in report
    )
    assert "Earliest destructive saliency transformation: grad_lost" in report
    assert "Raw/production parity check: PASS" in report


def test_candidate_with_only_sub_tolerance_zero_noise_passes(capsys):
    reference = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    candidate = torch.full_like(reference, 1e-17)
    stream_network = SimpleNamespace(
        saliency_diagnostic_maps={
            name: candidate for name in ("raw", "grad_lost", "ownership", "production")
        }
    )

    results = compare_saliency_candidates(
        stream_network, reference, rtol=1e-7, atol=1e-9
    )

    assert all(results.values())
    report = capsys.readouterr().out
    assert "exact maximum absolute error: 1.00000000000000007e-17" in report
    assert report.count("all tolerance-mismatch counts are zero") == 4
    assert "Earliest destructive saliency transformation: none" in report


def test_verbose_candidate_summary_includes_complete_mismatch_arrays(capsys):
    reference = torch.zeros((1, 1, 2, 3))
    divergent = reference.clone()
    divergent[0, 0, 1, 2] = 1.0
    stream_network = SimpleNamespace(
        saliency_diagnostic_maps={
            "raw": reference,
            "grad_lost": divergent,
            "ownership": divergent,
            "production": reference,
        }
    )

    compare_saliency_candidates(
        stream_network, reference, rtol=0, atol=1e-6, verbose=True
    )

    report = capsys.readouterr().out
    assert "per-row tolerance-mismatch counts: [0, 1]" in report
    assert "per-column tolerance-mismatch counts: [0, 0, 1]" in report


def test_production_disagreement_fails_the_regression(capsys):
    reference = torch.ones((1, 1, 1, 1), dtype=torch.float64)
    stream_network = SimpleNamespace(
        saliency_diagnostic_maps={
            "raw": reference.clone(),
            "grad_lost": reference + 1.0,
            "ownership": reference + 1.0,
            "production": reference + 1e-3,
        }
    )

    with pytest.raises(AssertionError, match="production"):
        compare_saliency_candidates(stream_network, reference, rtol=0, atol=1e-6)

    report = capsys.readouterr().out
    assert "Saliency candidate production:" in report
    assert "tolerance check (rtol=0.000e+00, atol=1.000e-06): FAIL" in report
    assert "Earliest destructive saliency transformation: grad_lost" in report


def test_raw_and_production_must_match_each_other(capsys):
    reference = torch.tensor([[[[100.0]]]], dtype=torch.float64)
    stream_network = SimpleNamespace(
        saliency_diagnostic_maps={
            "raw": reference - 9.0,
            "grad_lost": reference.clone(),
            "ownership": reference.clone(),
            "production": reference + 9.0,
        }
    )

    with pytest.raises(AssertionError, match="raw and production"):
        compare_saliency_candidates(stream_network, reference, rtol=0.1, atol=0)

    assert "Raw/production parity check: FAIL" in capsys.readouterr().out


def test_missing_additive_writes_are_reported_by_boundary_and_sides(capsys):
    reference = torch.tensor([[[[2.0, 2.0]]]])
    raw = reference.clone()
    cropped = torch.tensor([[[[1.0, 2.0]]]])
    count_maps = {
        "raw": torch.tensor([[[[2, 1]]]], dtype=torch.int32),
        "grad_lost": torch.ones((1, 1, 1, 2), dtype=torch.int32),
        "ownership": torch.ones((1, 1, 1, 2), dtype=torch.int32),
    }
    stream_network = SimpleNamespace(
        saliency_diagnostic_maps={
            "raw": raw,
            "grad_lost": cropped,
            "ownership": cropped,
            "production": raw,
        },
        saliency_diagnostic_write_count_maps=count_maps,
        saliency_diagnostic_records=[
            {
                "candidate_destination_slices": {
                    "grad_lost": (0, 0, 1, 2),
                    "ownership": (0, 0, 1, 2),
                },
                "sides": {"top": True, "left": False, "bottom": False, "right": True},
            }
        ],
    )

    compare_saliency_candidates(stream_network, reference, rtol=0, atol=1e-6)

    report = capsys.readouterr().out
    assert (
        "error: total=1, coordinates=[(0, 0)], bounding box "
        "[top, left, bottom, right)=(0, 0, 1, 1)" in report
    )
    assert "boundary=(0, 0, 1, 2)" in report
    assert (
        "Sides(top=True, left=False, bottom=False, right=True): 1 coordinates" in report
    )


def test_missing_additive_write_coordinates_are_bounded_unless_verbose(capsys):
    width = 25
    reference = torch.full((1, 1, 1, width), 2.0)
    cropped = torch.ones_like(reference)
    counts = {
        "raw": torch.full_like(reference, 2, dtype=torch.int32),
        "grad_lost": torch.ones_like(reference, dtype=torch.int32),
        "ownership": torch.ones_like(reference, dtype=torch.int32),
    }
    stream_network = SimpleNamespace(
        saliency_diagnostic_maps={
            "raw": reference,
            "grad_lost": cropped,
            "ownership": cropped,
            "production": reference,
        },
        saliency_diagnostic_write_count_maps=counts,
        saliency_diagnostic_records=[],
    )

    compare_saliency_candidates(stream_network, reference, rtol=0, atol=1e-6)
    report = capsys.readouterr().out
    assert "total=25" in report
    assert "(0, 19)" in report
    assert "(0, 20)" not in report
    assert "... 5 additional coordinates omitted" in report
    assert "bounding box [top, left, bottom, right)=(0, 0, 1, 25)" in report

    compare_saliency_candidates(
        stream_network, reference, rtol=0, atol=1e-6, verbose=True
    )
    verbose_report = capsys.readouterr().out
    assert "(0, 24)" in verbose_report
    assert "additional coordinates omitted" not in verbose_report
