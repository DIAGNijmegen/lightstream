"""Compatibility boundary for the optional Neighborhood Attention dependency.

NAT support is deliberately not imported by :mod:`lightstream.models`, so the
rest of Lightstream remains usable without NATTEN.  Code constructing a NAT
backbone should obtain its attention operation through :func:`load_na2d`; this
verifies the one dependency pair
covered by the initial streaming target before importing NATTEN's native code.
"""

from .compat import (
    SUPPORTED_NATTEN_VERSION,
    SUPPORTED_TORCH_VERSION,
    NATTENCompatibilityError,
    load_na2d,
    require_supported_versions,
)

__all__ = [
    "SUPPORTED_NATTEN_VERSION",
    "SUPPORTED_TORCH_VERSION",
    "NATTENCompatibilityError",
    "load_na2d",
    "require_supported_versions",
]
