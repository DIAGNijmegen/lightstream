from types import SimpleNamespace

import pytest

from lightstream.models.nat import compat


def _versions(torch="2.13.0", natten="0.21.7"):
    return lambda package: {"torch": torch, "natten": natten}[package]


def test_supported_versions_accept_local_wheel_suffixes(monkeypatch):
    monkeypatch.setattr(
        compat,
        "version",
        _versions("2.13.0+cu132", "0.21.7+torch2130cu132"),
    )

    compat.require_supported_versions()


@pytest.mark.parametrize(
    ("torch_version", "natten_version"),
    [("2.14.0", "0.21.7"), ("2.13.0", "0.21.6")],
)
def test_incompatible_pair_has_clear_error(monkeypatch, torch_version, natten_version):
    monkeypatch.setattr(compat, "version", _versions(torch_version, natten_version))

    with pytest.raises(compat.NATTENCompatibilityError) as error:
        compat.require_supported_versions()

    message = str(error.value)
    assert f"torch=={torch_version}" in message
    assert f"natten=={natten_version}" in message
    assert "torch==2.13.0 and natten==0.21.7" in message


def test_missing_natten_explains_optional_install(monkeypatch):
    def missing_natten(package):
        raise compat.PackageNotFoundError(package)

    monkeypatch.setattr(compat, "version", missing_natten)

    with pytest.raises(ImportError, match=r"lightstream\[nat\]"):
        compat.require_supported_versions()


def test_loader_uses_functional_api(monkeypatch):
    na2d = object()
    monkeypatch.setattr(compat, "version", _versions())
    monkeypatch.setattr(
        compat,
        "import_module",
        lambda name: SimpleNamespace(na2d=na2d),
    )

    assert compat.load_na2d() is na2d


def test_loader_rejects_missing_functional_api(monkeypatch):
    monkeypatch.setattr(compat, "version", _versions())
    monkeypatch.setattr(compat, "import_module", lambda name: SimpleNamespace())

    with pytest.raises(
        compat.NATTENCompatibilityError,
        match=r"natten\.functional\.na2d",
    ):
        compat.load_na2d()
