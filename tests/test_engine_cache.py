import pytest

from lightstream.core.engine.cache import (
    CACHE_VERSION,
    PlanCacheError,
    load_plan,
    serialize_plan,
)
from lightstream.core.engine.configuration import HeadPlan, ModulePlan, StreamingPlan, TilePlan
from lightstream.core.engine.geometry import Lost
from lightstream.core.engine.reducers import StaticReducerBinding


def _plan():
    return StreamingPlan(
        tile=TilePlan((1, 3, 16, 16), Lost(1, 1, 1, 1), (2, 2)),
        heads=(HeadPlan((1, 8, 8, 8), (1, 2, 2), Lost(0, 1, 0, 1)),),
        modules=(ModulePlan("conv", "Conv2d", (("stride", (1, 2, 2)),)),),
        reducer_heads=(StaticReducerBinding("pool", "MeanReducer"),),
        output_structure=("dict", [("features", ("tensor", None))]),
    )


def test_plan_cache_round_trip_contains_only_configuration():
    expected = _plan()
    cache = serialize_plan(expected)

    assert cache["version"] == CACHE_VERSION
    assert "session" not in repr(cache).lower()
    assert load_plan(cache, expected=expected) == expected


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cache: cache.update(version=999), "unsupported format version"),
        (lambda cache: cache["tile"].update(input_shape=[1, 3, 0, 16]), "positive integers"),
        (lambda cache: cache["modules"][0].update(module_type=42), "identity"),
        (lambda cache: cache["heads"][0].update(stride=[1, 2]), "3 dimensions"),
        (lambda cache: cache.update(output_structure={"__type__": "tuple", "values": ["bad"]}), "output specification"),
        (lambda cache: cache["reducer_bindings"][0].update(reducer_type=None), "identity"),
    ],
)
def test_plan_cache_rejects_invalid_configuration(mutate, message):
    cache = serialize_plan(_plan())
    mutate(cache)
    with pytest.raises(PlanCacheError, match=message):
        load_plan(cache)


def test_plan_cache_validates_current_identities():
    cache = serialize_plan(_plan())
    cache["modules"][0]["module_type"] = "Linear"

    with pytest.raises(PlanCacheError, match="module identities mismatch"):
        load_plan(cache, expected=_plan())


def test_legacy_cache_requires_explicit_safe_migration_context():
    with pytest.raises(PlanCacheError, match="legacy dictionary format.*regenerate"):
        load_plan({"net_stats": {}})
