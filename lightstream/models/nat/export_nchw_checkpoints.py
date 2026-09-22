"""Export official NAT checkpoints in Lightstream's NCHW layout.

The exported ``state_dict`` retains the original linear classifier.  The
current :class:`~lightstream.models.nat.nchw.NCHWNAT` factories load only
``backbone_state_dict`` because they produce an NCHW feature map.  To
reconstruct classification logits, globally average that map over its two
spatial dimensions and apply the preserved ``head.weight`` and ``head.bias``
as a linear layer.

For example, export one checkpoint with::

    python -m lightstream.models.nat.export_nchw_checkpoints \
        --variant nat_mini --output nat_mini-nchw.pth

Or export all official variants with::

    python -m lightstream.models.nat.export_nchw_checkpoints \
        --all --output-dir release-checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from lightstream.models.nat import nat as nhwc_nat
from lightstream.models.nat import nchw as nchw_nat
from lightstream.models.nat.nchw import convert_nhwc_nat_state_dict


FORMAT = "lightstream-nchw-nat-v1"
EXPECTED_CLASSIFIER_KEYS = frozenset({"head.weight", "head.bias"})

# Keep the correspondence explicit: an accidental alias or synthetic variant
# should never silently become part of a release export.
VARIANTS: dict[
    str, tuple[Callable[..., nn.Module], Callable[..., nn.Module], str]
] = {
    "nat_mini": (
        nhwc_nat.nat_mini,
        nchw_nat.nchw_nat_mini,
        "nat_mini_1k",
    ),
    "nat_tiny": (
        nhwc_nat.nat_tiny,
        nchw_nat.nchw_nat_tiny,
        "nat_tiny_1k",
    ),
    "nat_small": (
        nhwc_nat.nat_small,
        nchw_nat.nchw_nat_small,
        "nat_small_1k",
    ),
    "nat_base": (
        nhwc_nat.nat_base,
        nchw_nat.nchw_nat_base,
        "nat_base_1k",
    ),
}


def _cpu_state_dict(state_dict):
    """Clone a state dict as detached CPU tensors without retaining a model."""

    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _verify_reloaded_checkpoint(
    output: Path, nchw_factory: Callable[..., nn.Module]
) -> None:
    """Reload an export through the safe weights-only path and verify tensors."""

    payload = torch.load(output, map_location="cpu", weights_only=True)
    exported = payload["backbone_state_dict"]
    fresh_model = nchw_factory(pretrained=False)
    fresh_model.load_state_dict(exported, strict=True)
    reloaded = fresh_model.state_dict()
    if reloaded.keys() != exported.keys():
        raise RuntimeError("reloaded backbone keys differ from the exported keys")
    unequal = [
        key for key in exported if not torch.equal(reloaded[key].cpu(), exported[key])
    ]
    if unequal:
        raise RuntimeError(f"reloaded backbone tensors differ: {unequal}")


def export_variant(variant: str, output: str | Path) -> Path:
    """Download, convert, validate, and save one official NAT variant."""

    try:
        nhwc_factory, nchw_factory, checkpoint_id = VARIANTS[variant]
    except KeyError as error:
        supported = ", ".join(VARIANTS)
        raise ValueError(f"unknown variant {variant!r}; choose one of: {supported}") from error

    original_model = nhwc_factory(pretrained=True)
    original_state = original_model.state_dict()
    missing_head = EXPECTED_CLASSIFIER_KEYS.difference(original_state)
    if missing_head:
        raise RuntimeError(f"upstream checkpoint is missing classifier keys: {missing_head}")
    converted = convert_nhwc_nat_state_dict(original_state)

    nchw_model = nchw_factory(pretrained=False)
    model_keys = set(nchw_model.state_dict())
    backbone_state = {
        key: converted[key] for key in converted.keys() if key in model_keys
    }
    if set(backbone_state) != model_keys:
        missing = model_keys.difference(backbone_state)
        extra = set(backbone_state).difference(model_keys)
        raise RuntimeError(f"backbone key mismatch; missing={missing}, extra={extra}")
    nchw_model.load_state_dict(backbone_state, strict=True)

    remaining_keys = set(converted).difference(backbone_state)
    if remaining_keys != EXPECTED_CLASSIFIER_KEYS:
        raise RuntimeError(
            "converted non-backbone keys differ from the expected classifier keys; "
            f"found={remaining_keys}, expected={set(EXPECTED_CLASSIFIER_KEYS)}"
        )

    complete_cpu_state = _cpu_state_dict(converted)
    # Save the post-strict-load model state, rather than merely repeating the
    # selected input mapping, so this field records precisely what was tested.
    backbone_cpu_state = _cpu_state_dict(nchw_model.state_dict())
    checkpoint_url = nhwc_nat.model_urls[checkpoint_id]
    payload = {
        "state_dict": complete_cpu_state,
        "backbone_state_dict": backbone_cpu_state,
        "variant": variant,
        "checkpoint_id": checkpoint_id,
        "checkpoint_url": checkpoint_url,
        "format": FORMAT,
        "num_classes": int(original_model.num_classes),
        "num_features": int(original_model.num_features),
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    _verify_reloaded_checkpoint(output, nchw_factory)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    size = output.stat().st_size
    print(f"output: {output}")
    print(f"size: {size} bytes")
    print(f"state-dict keys: {len(complete_cpu_state)}")
    print(f"head.weight preserved: {'head.weight' in complete_cpu_state}")
    print(f"head.bias preserved: {'head.bias' in complete_cpu_state}")
    print(f"sha256: {digest}")
    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official pretrained NAT checkpoints for NCHWNAT.",
        epilog=(
            "NCHWNAT loads only backbone_state_dict. For classification, global-"
            "average-pool the NCHW feature map and apply the preserved linear head."
        ),
    )
    parser.add_argument("--variant", choices=VARIANTS, default="nat_mini")
    parser.add_argument("--output", type=Path, help="destination .pth file")
    parser.add_argument("--all", action="store_true", help="export every variant")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="directory used by --all"
    )
    args = parser.parse_args(argv)
    if args.all and args.output is not None:
        parser.error("--output cannot be combined with --all; use --output-dir")
    if not args.all and args.output_dir != Path("."):
        parser.error("--output-dir requires --all")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the checkpoint exporter command-line interface."""

    args = _parse_args(argv)
    if args.all:
        for variant in VARIANTS:
            export_variant(variant, args.output_dir / f"{variant}-nchw.pth")
    else:
        output = args.output or Path(f"{args.variant}-nchw.pth")
        export_variant(args.variant, output)


if __name__ == "__main__":
    main()
