from types import SimpleNamespace

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
            "production": grad_lost,
        }
    )

    results = compare_saliency_candidates(
        stream_network, reference, rtol=0.0, atol=1e-6
    )

    assert results == {
        "raw": True,
        "grad_lost": False,
        "ownership": False,
        "production": False,
    }
    report = capsys.readouterr().out
    assert "error bounding box [top, left, bottom, right): (1, 2, 2, 3)" in report
    assert "per-row tolerance-mismatch counts: [0, 1]" in report
    assert "per-column tolerance-mismatch counts: [0, 0, 1]" in report
    assert "Earliest failing saliency transformation: grad_lost" in report


def test_candidate_with_only_sub_tolerance_zero_noise_passes(capsys):
    reference = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    candidate = torch.full_like(reference, 1e-17)
    stream_network = SimpleNamespace(
        saliency_diagnostic_maps={
            name: candidate
            for name in ("raw", "grad_lost", "ownership", "production")
        }
    )

    results = compare_saliency_candidates(
        stream_network, reference, rtol=1e-7, atol=1e-9
    )

    assert all(results.values())
    report = capsys.readouterr().out
    assert "exact maximum absolute error: 1.00000000000000007e-17" in report
    assert "Earliest failing saliency transformation: none" in report


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
        saliency_diagnostic_records=[{
            "candidate_destination_slices": {
                "grad_lost": (0, 0, 1, 2),
                "ownership": (0, 0, 1, 2),
            },
            "sides": {"top": True, "left": False, "bottom": False, "right": True},
        }],
    )

    compare_saliency_candidates(stream_network, reference, rtol=0, atol=1e-6)

    report = capsys.readouterr().out
    assert "error: [(0, 0)]" in report
    assert "boundary=(0, 0, 1, 2)" in report
    assert "Sides(top=True, left=False, bottom=False, right=True): 1 coordinates" in report
