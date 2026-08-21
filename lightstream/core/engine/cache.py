"""Versioned, validation-first persistence for :mod:`lightstream` plans.

The cache deliberately contains only the immutable ``StreamingPlan``.  In
particular, executor sessions and reducer instances must never cross this
boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .configuration import HeadPlan, ModulePlan, StreamingPlan, TilePlan
from .geometry import Lost
from .reducers import StaticReducerBinding

CACHE_FORMAT = "lightstream.streaming-plan"
CACHE_VERSION = 1


class PlanCacheError(ValueError):
    """A plan cache is malformed, stale, or cannot be migrated safely."""


def _error(message: str) -> PlanCacheError:
    return PlanCacheError(f"Invalid streaming plan cache: {message}")


def _lost(value: Any, field: str) -> Lost:
    if isinstance(value, Lost):
        values = (value.top, value.left, value.bottom, value.right)
    elif isinstance(value, Mapping):
        try:
            values = tuple(value[key] for key in ("top", "left", "bottom", "right"))
        except KeyError as exc:
            raise _error(f"{field} is missing {exc.args[0]!r}") from exc
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        values = tuple(value)
    else:
        raise _error(f"{field} must contain top, left, bottom, and right")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values):
        raise _error(f"{field} values must be non-negative integers")
    return Lost(*values)


def _shape(value: Any, field: str, *, dimensions: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error(f"{field} must be an integer sequence")
    result = tuple(value)
    if dimensions is not None and len(result) != dimensions:
        raise _error(f"{field} must have {dimensions} dimensions, got {len(result)}")
    if not result or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in result):
        raise _error(f"{field} values must be positive integers")
    return result


def _value_to_data(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Lost):
        return {"__type__": "lost", "values": [value.top, value.left, value.bottom, value.right]}
    if isinstance(value, torch.Tensor):
        return {"__type__": "tensor", "dtype": str(value.dtype), "values": value.detach().cpu().tolist()}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "values": [_value_to_data(item) for item in value]}
    if isinstance(value, list):
        return {"__type__": "list", "values": [_value_to_data(item) for item in value]}
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {"__type__": "dict", "values": {key: _value_to_data(item) for key, item in value.items()}}
    raise TypeError(f"Unsupported immutable plan value: {type(value).__module__}.{type(value).__qualname__}")


def _value_from_data(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if not isinstance(value, Mapping) or set(value) != {"__type__", "values"} and set(value) != {"__type__", "dtype", "values"}:
        raise _error(f"{field} contains an unsupported encoded value")
    kind = value["__type__"]
    if kind == "lost":
        return _lost(value["values"], field)
    if kind in {"tuple", "list"}:
        if not isinstance(value["values"], list):
            raise _error(f"{field} {kind} payload must be a list")
        decoded = [_value_from_data(item, field) for item in value["values"]]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict":
        if not isinstance(value["values"], Mapping) or not all(isinstance(key, str) for key in value["values"]):
            raise _error(f"{field} dictionary payload must have string keys")
        return {key: _value_from_data(item, field) for key, item in value["values"].items()}
    if kind == "tensor":
        dtype_name = value.get("dtype")
        if not isinstance(dtype_name, str) or not dtype_name.startswith("torch.") or not hasattr(torch, dtype_name[6:]):
            raise _error(f"{field} has unsupported tensor dtype {dtype_name!r}")
        return torch.tensor(value["values"], dtype=getattr(torch, dtype_name[6:]))
    raise _error(f"{field} has unsupported encoded type {kind!r}")


def serialize_plan(plan: StreamingPlan) -> dict[str, Any]:
    """Return a portable dictionary containing immutable plan configuration."""
    if not isinstance(plan, StreamingPlan):
        raise TypeError("serialize_plan expects a StreamingPlan")
    return {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "tile": {"input_shape": list(plan.tile.input_shape), "gradient_loss": _value_to_data(plan.tile.gradient_loss),
                 "internal_alignment": list(plan.tile.internal_alignment)},
        "heads": [{"tile_output_shape": list(head.tile_output_shape), "stride": list(head.stride),
                   "loss": _value_to_data(head.loss)} for head in plan.heads],
        "modules": [{"name": module.name, "module_type": module.module_type,
                     "statistics": [[key, _value_to_data(value)] for key, value in module.statistics]}
                    for module in plan.modules],
        "reducer_bindings": [{"name": binding.name, "reducer_type": binding.reducer_type}
                             for binding in plan.reducer_heads],
        "output_structure": _value_to_data(plan.output_structure),
    }


def _require_keys(value: Any, keys: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be a dictionary")
    missing, extra = keys - set(value), set(value) - keys
    if missing or extra:
        raise _error(f"{field} keys differ (missing={sorted(missing)}, extra={sorted(extra)})")
    return value


def load_plan(data: Mapping[str, Any], *, expected: StreamingPlan | None = None,
              expected_tile_shape: Sequence[int] | None = None) -> StreamingPlan:
    """Decode and validate a cache, optionally checking it against a current plan.

    Legacy SCNN dictionaries are accepted only when ``expected`` is supplied;
    use :func:`migrate_legacy_cache` to make that compatibility path explicit.
    """
    if not isinstance(data, Mapping):
        raise _error("root must be a dictionary")
    if "format" not in data or "version" not in data:
        return migrate_legacy_cache(data, expected=expected, expected_tile_shape=expected_tile_shape)
    root = _require_keys(data, {"format", "version", "tile", "heads", "modules", "reducer_bindings", "output_structure"}, "root")
    if root["format"] != CACHE_FORMAT:
        raise _error(f"unsupported format {root['format']!r}; regenerate the cache")
    if root["version"] != CACHE_VERSION:
        raise _error(f"unsupported format version {root['version']!r} (supported: {CACHE_VERSION}); regenerate the cache")
    tile_data = _require_keys(root["tile"], {"input_shape", "gradient_loss", "internal_alignment"}, "tile")
    tile_shape = _shape(tile_data["input_shape"], "tile.input_shape", dimensions=4)
    alignment = _shape(tile_data["internal_alignment"], "tile.internal_alignment", dimensions=2)
    tile = TilePlan(tile_shape, _lost(_value_from_data(tile_data["gradient_loss"], "tile.gradient_loss"), "tile.gradient_loss"), alignment)

    if not isinstance(root["heads"], list) or not root["heads"]:
        raise _error("heads must be a non-empty list")
    heads = []
    for index, raw in enumerate(root["heads"]):
        item = _require_keys(raw, {"tile_output_shape", "stride", "loss"}, f"heads[{index}]")
        shape = _shape(item["tile_output_shape"], f"heads[{index}].tile_output_shape", dimensions=4)
        stride = _shape(item["stride"], f"heads[{index}].stride", dimensions=3)
        heads.append(HeadPlan(shape, stride, _lost(_value_from_data(item["loss"], f"heads[{index}].loss"), f"heads[{index}].loss")))

    modules = []
    if not isinstance(root["modules"], list): raise _error("modules must be a list")
    for index, raw in enumerate(root["modules"]):
        item = _require_keys(raw, {"name", "module_type", "statistics"}, f"modules[{index}]")
        if not isinstance(item["name"], str) or not isinstance(item["module_type"], str): raise _error(f"modules[{index}] identity must contain strings")
        if not isinstance(item["statistics"], list): raise _error(f"modules[{index}].statistics must be a list")
        stats = []
        for pair in item["statistics"]:
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str): raise _error(f"modules[{index}].statistics entries must be [name, value]")
            stats.append((pair[0], _value_from_data(pair[1], f"modules[{index}].statistics.{pair[0]}")))
        if tuple(key for key, _ in stats) != tuple(sorted(key for key, _ in stats)): raise _error(f"modules[{index}].statistics must be sorted")
        modules.append(ModulePlan(item["name"], item["module_type"], tuple(stats)))

    reducers = []
    if not isinstance(root["reducer_bindings"], list): raise _error("reducer_bindings must be a list")
    for index, raw in enumerate(root["reducer_bindings"]):
        item = _require_keys(raw, {"name", "reducer_type"}, f"reducer_bindings[{index}]")
        if not isinstance(item["name"], str) or not isinstance(item["reducer_type"], str): raise _error(f"reducer_bindings[{index}] identity must contain strings")
        reducers.append(StaticReducerBinding(item["name"], item["reducer_type"]))
    output = _value_from_data(root["output_structure"], "output_structure")
    _validate_output_structure(output)
    plan = StreamingPlan(tile, tuple(heads), tuple(modules), tuple(reducers), output)
    _validate_expected(plan, expected, expected_tile_shape)
    return plan


def _validate_output_structure(spec: Any, field: str = "output_structure") -> None:
    if not isinstance(spec, tuple) or len(spec) != 2 or spec[0] not in {"tensor", "tuple", "list", "dict"}:
        raise _error(f"{field} is not a valid output specification")
    kind, children = spec
    if kind == "tensor":
        if children is not None: raise _error(f"{field} tensor payload must be None")
    elif kind in {"tuple", "list"}:
        if not isinstance(children, (tuple, list)): raise _error(f"{field} children must be a sequence")
        for index, child in enumerate(children): _validate_output_structure(child, f"{field}[{index}]")
    else:
        if not isinstance(children, (tuple, list)): raise _error(f"{field} dictionary children must be a sequence")
        for index, child in enumerate(children):
            if not isinstance(child, (tuple, list)) or len(child) != 2: raise _error(f"{field}[{index}] must contain a key and child")
            _validate_output_structure(child[1], f"{field}[{child[0]!r}]")


def _validate_expected(plan: StreamingPlan, expected: StreamingPlan | None, expected_tile_shape: Sequence[int] | None) -> None:
    target_shape = tuple(expected_tile_shape) if expected_tile_shape is not None else (expected.tile.input_shape if expected else None)
    if target_shape is not None and plan.tile.input_shape != tuple(target_shape): raise _error(f"tile shape mismatch: cached={plan.tile.input_shape}, expected={tuple(target_shape)}")
    if expected is None: return
    checks = (("module identities", tuple((x.name, x.module_type) for x in plan.modules), tuple((x.name, x.module_type) for x in expected.modules)),
              ("head metadata", plan.heads, expected.heads), ("output structure", plan.output_structure, expected.output_structure),
              ("reducer bindings", plan.reducer_heads, expected.reducer_heads))
    for label, cached, wanted in checks:
        if cached != wanted: raise _error(f"{label} mismatch: cached={cached!r}, expected={wanted!r}")


def migrate_legacy_cache(data: Mapping[str, Any], *, expected: StreamingPlan | None = None,
                         expected_tile_shape: Sequence[int] | None = None) -> StreamingPlan:
    """Validate a legacy SCNN dictionary and return its equivalent current plan."""
    if expected is None:
        raise _error("legacy dictionary format lacks module/reducer identities; regenerate the cache or migrate it with expected=current_plan")
    required = {"net_stats", "tile_gradient_lost", "tile_output_shape", "output_stride", "tile_output_lost"}
    missing = required - set(data)
    if missing: raise _error(f"legacy dictionary is missing {sorted(missing)}; regenerate the cache")
    shapes = data.get("tile_output_shapes", [data["tile_output_shape"]])
    strides = data.get("output_stride_per_output", [data["output_stride"]])
    losses = data.get("tile_output_lost_all", [data["tile_output_lost"]])
    if not (len(shapes) == len(strides) == len(losses)): raise _error("legacy head metadata lengths differ; regenerate the cache")
    stats = data["net_stats"]
    if not isinstance(stats, Mapping): raise _error("legacy net_stats must be a dictionary")
    modules = tuple(ModulePlan(module.name, module.module_type, tuple(sorted(stats.get(module.name, {}).items()))) for module in expected.modules)
    plan = StreamingPlan(TilePlan(expected.tile.input_shape, _lost(data["tile_gradient_lost"], "tile_gradient_lost"), expected.tile.internal_alignment),
                         tuple(HeadPlan(_shape(shape, "tile_output_shape", dimensions=4), _shape(tuple(int(x) for x in stride), "output_stride", dimensions=3), _lost(loss, "tile_output_lost")) for shape, stride, loss in zip(shapes, strides, losses)),
                         modules, expected.reducer_heads, data.get("output_spec", ("tensor", None)))
    _validate_output_structure(plan.output_structure)
    _validate_expected(plan, expected, expected_tile_shape)
    return plan
