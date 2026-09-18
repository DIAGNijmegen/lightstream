"""Version and API checks for Lightstream's initial NAT integration."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

SUPPORTED_TORCH_VERSION = "2.13.0"
SUPPORTED_NATTEN_VERSION = "0.21.7"


class NATTENCompatibilityError(RuntimeError):
    """Raised when the installed PyTorch/NATTEN pair is unsupported."""


def _release(version_string: str) -> str:
    """Discard a wheel's local suffix (for example ``+cu118``)."""

    return version_string.split("+", maxsplit=1)[0]


def require_supported_versions() -> None:
    """Validate the only PyTorch/NATTEN pair supported by the NAT target.

    Raises
    ------
    ImportError
        If NATTEN is not installed.
    NATTENCompatibilityError
        If either installed release differs from the supported pair.
    """

    try:
        installed_natten = version("natten")
    except PackageNotFoundError as error:
        raise ImportError(
            "Lightstream NAT support is optional. Install the supported "
            "dependencies with `pip install 'lightstream[nat]'`."
        ) from error

    installed_torch = version("torch")
    actual_pair = (_release(installed_torch), _release(installed_natten))
    expected_pair = (SUPPORTED_TORCH_VERSION, SUPPORTED_NATTEN_VERSION)
    if actual_pair != expected_pair:
        raise NATTENCompatibilityError(
            "Unsupported PyTorch/NATTEN versions for Lightstream NAT: found "
            f"torch=={installed_torch} and natten=={installed_natten}; expected "
            f"torch=={SUPPORTED_TORCH_VERSION} and "
            f"natten=={SUPPORTED_NATTEN_VERSION}. Install the supported pair "
            "with `pip install 'lightstream[nat]'`."
        )


def load_na2d() -> Any:
    """Return NATTEN 0.21.7's functional ``na2d`` operation."""

    require_supported_versions()
    module = import_module("natten.functional")
    try:
        return module.na2d
    except AttributeError as error:
        raise NATTENCompatibilityError(
            "natten==0.21.7 must provide the functional "
            "`natten.functional.na2d` API, but that symbol is missing."
        ) from error
