import importlib.util
from pathlib import Path


_UTILS_PATH = Path(__file__).resolve().parents[1] / "lightstream" / "core" / "scnn" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("scnn_utils", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

Box = _MODULE.Box
_new_value_indices = _MODULE._new_value_indices


def test_new_value_indices_accepts_exact_border_progression() -> None:
    old = Box(y=0, height=100, x=10, width=0, sides=None)
    data = Box(y=0, height=0, x=10, width=0, sides=None)

    new_box, updated = _new_value_indices((1, 1, 10, 10), data, old)

    assert new_box.x == 0
    assert new_box.width == 10
    assert updated.x == 20


def test_new_value_indices_rejects_forward_jump_assertion() -> None:
    old = Box(y=0, height=100, x=10, width=0, sides=None)
    data = Box(y=0, height=0, x=20, width=0, sides=None)

    try:
        _new_value_indices((1, 1, 10, 10), data, old)
    except AssertionError as exc:
        assert "Misses data in x-axis" in str(exc)
    else:
        raise AssertionError("Expected assertion for large forward jump")


def test_new_value_indices_tolerates_tiny_forward_drift() -> None:
    old = Box(y=0, height=100, x=10, width=0, sides=None)
    data = Box(y=1, height=0, x=11, width=0, sides=None)

    new_box, updated = _new_value_indices((1, 1, 10, 10), data, old)

    assert new_box.x == 0
    assert new_box.y == 0
    assert new_box.width == 10
    assert new_box.height == 10
    assert updated.x == 20
