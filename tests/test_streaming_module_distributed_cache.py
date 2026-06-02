import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch.nn as nn


def _load_streaming_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "lightstream"
        / "modules"
        / "streaming.py"
    )
    stub_names = [
        "lightstream",
        "lightstream.core",
        "lightstream.core.constructor",
        "lightstream.core.reducer",
    ]
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    lightstream = types.ModuleType("lightstream")
    lightstream.__path__ = []
    lightstream_core = types.ModuleType("lightstream.core")
    lightstream_core.__path__ = []
    constructor_module = types.ModuleType("lightstream.core.constructor")
    constructor_module.StreamingConstructor = None
    reducer_module = types.ModuleType("lightstream.core.reducer")
    reducer_module.BaseReducer = nn.Module

    sys.modules["lightstream"] = lightstream
    sys.modules["lightstream.core"] = lightstream_core
    sys.modules["lightstream.core.constructor"] = constructor_module
    sys.modules["lightstream.core.reducer"] = reducer_module

    try:
        spec = importlib.util.spec_from_file_location(
            "_streaming_module_under_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module

    return module


streaming_module = _load_streaming_module()
StreamingModule = streaming_module.StreamingModule


class DummyPreparedNetwork:
    def get_tile_cache(self):
        return {"computed": True}


class DummyConstructor:
    calls = []

    def __init__(self, stream_network, tile_size, tile_cache=None, **kwargs):
        self.copy_to_gpu = kwargs.get("copy_to_gpu", False)
        self.tile_cache = tile_cache
        DummyConstructor.calls.append(
            {
                "stream_network": stream_network,
                "tile_size": tile_size,
                "tile_cache": tile_cache,
                "kwargs": kwargs,
            }
        )

    def prepare_streaming_model(self):
        return DummyPreparedNetwork()


@pytest.fixture(autouse=True)
def _reset_dummy_constructor(monkeypatch):
    DummyConstructor.calls = []
    monkeypatch.setattr(streaming_module, "StreamingConstructor", DummyConstructor)
    monkeypatch.setattr(
        StreamingModule,
        "build_tile_cache_metadata",
        lambda self, stream_network: {},
    )


def test_distributed_rank_zero_computes_and_saves_missing_cache(monkeypatch, tmp_path):
    saves = []
    barriers = []

    monkeypatch.setattr(StreamingModule, "load_tile_cache_if_needed", lambda self: None)
    monkeypatch.setattr(
        StreamingModule,
        "save_tile_cache_if_needed",
        lambda self, overwrite=False: saves.append(overwrite),
    )
    monkeypatch.setattr(StreamingModule, "_distributed_is_initialized", staticmethod(lambda: True))
    monkeypatch.setattr(StreamingModule, "_distributed_world_size", staticmethod(lambda: 2))
    monkeypatch.setattr(StreamingModule, "_distributed_rank", staticmethod(lambda: 0))
    monkeypatch.setattr(
        StreamingModule,
        "_distributed_barrier",
        staticmethod(lambda: barriers.append("barrier")),
    )

    StreamingModule(nn.Identity(), 8, tile_cache_path=tmp_path / "cache", copy_to_gpu=True)

    assert [call["tile_cache"] for call in DummyConstructor.calls] == [None]
    assert DummyConstructor.calls[0]["kwargs"] == {"copy_to_gpu": True}
    assert saves == [False]
    assert barriers == ["barrier"]


def test_distributed_rank_zero_overwrites_stale_cache(monkeypatch, tmp_path):
    saves = []
    barriers = []

    def load_stale_cache(self):
        self._tile_cache_was_ignored = True
        return None

    monkeypatch.setattr(StreamingModule, "load_tile_cache_if_needed", load_stale_cache)
    monkeypatch.setattr(
        StreamingModule,
        "save_tile_cache_if_needed",
        lambda self, overwrite=False: saves.append(overwrite),
    )
    monkeypatch.setattr(StreamingModule, "_distributed_is_initialized", staticmethod(lambda: True))
    monkeypatch.setattr(StreamingModule, "_distributed_world_size", staticmethod(lambda: 2))
    monkeypatch.setattr(StreamingModule, "_distributed_rank", staticmethod(lambda: 0))
    monkeypatch.setattr(
        StreamingModule,
        "_distributed_barrier",
        staticmethod(lambda: barriers.append("barrier")),
    )

    StreamingModule(nn.Identity(), 8, tile_cache_path=tmp_path / "cache")

    assert [call["tile_cache"] for call in DummyConstructor.calls] == [None]
    assert saves == [True]
    assert barriers == ["barrier"]


def test_distributed_rank_zero_reloads_cache_after_lock(monkeypatch, tmp_path):
    loads = iter([None, {"cached": True}])
    saves = []
    barriers = []
    lock_events = []

    class RecordingLock:
        def __enter__(self):
            lock_events.append("acquire")

        def __exit__(self, exc_type, exc, tb):
            lock_events.append("release")

    monkeypatch.setattr(StreamingModule, "load_tile_cache_if_needed", lambda self: next(loads))
    monkeypatch.setattr(
        StreamingModule,
        "save_tile_cache_if_needed",
        lambda self, overwrite=False: saves.append(overwrite),
    )
    monkeypatch.setattr(StreamingModule, "_exclusive_tile_cache_lock", lambda self: RecordingLock())
    monkeypatch.setattr(StreamingModule, "_distributed_is_initialized", staticmethod(lambda: True))
    monkeypatch.setattr(StreamingModule, "_distributed_world_size", staticmethod(lambda: 2))
    monkeypatch.setattr(StreamingModule, "_distributed_rank", staticmethod(lambda: 0))
    monkeypatch.setattr(
        StreamingModule,
        "_distributed_barrier",
        staticmethod(lambda: barriers.append("barrier")),
    )

    StreamingModule(nn.Identity(), 8, tile_cache_path=tmp_path / "cache")

    assert [call["tile_cache"] for call in DummyConstructor.calls] == [{"cached": True}]
    assert saves == []
    assert lock_events == ["acquire", "release"]
    assert barriers == ["barrier"]


def test_tile_cache_lock_path_is_next_to_cache_file(tmp_path):
    module = StreamingModule.__new__(StreamingModule)
    module.tile_cache_dir = tmp_path
    module.tile_cache_fname = "cache"

    assert module._tile_cache_lock_location() == tmp_path / "cache.lock"


def test_distributed_nonzero_rank_waits_then_loads_cache(monkeypatch, tmp_path):
    loads = iter([None, {"cached": True}])
    saves = []
    barriers = []

    monkeypatch.setattr(StreamingModule, "load_tile_cache_if_needed", lambda self: next(loads))
    monkeypatch.setattr(
        StreamingModule,
        "save_tile_cache_if_needed",
        lambda self, overwrite=False: saves.append(overwrite),
    )
    monkeypatch.setattr(StreamingModule, "_distributed_is_initialized", staticmethod(lambda: True))
    monkeypatch.setattr(StreamingModule, "_distributed_world_size", staticmethod(lambda: 2))
    monkeypatch.setattr(StreamingModule, "_distributed_rank", staticmethod(lambda: 1))
    monkeypatch.setattr(
        StreamingModule,
        "_distributed_barrier",
        staticmethod(lambda: barriers.append("barrier")),
    )

    StreamingModule(nn.Identity(), 8, tile_cache_path=tmp_path / "cache")

    assert [call["tile_cache"] for call in DummyConstructor.calls] == [{"cached": True}]
    assert saves == []
    assert barriers == ["barrier"]


def test_distributed_nonzero_rank_errors_when_cache_missing_after_barrier(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache"

    monkeypatch.setattr(StreamingModule, "load_tile_cache_if_needed", lambda self: None)
    monkeypatch.setattr(
        StreamingModule,
        "save_tile_cache_if_needed",
        lambda self, overwrite=False: None,
    )
    monkeypatch.setattr(StreamingModule, "_distributed_is_initialized", staticmethod(lambda: True))
    monkeypatch.setattr(StreamingModule, "_distributed_world_size", staticmethod(lambda: 2))
    monkeypatch.setattr(StreamingModule, "_distributed_rank", staticmethod(lambda: 1))
    monkeypatch.setattr(StreamingModule, "_distributed_barrier", staticmethod(lambda: None))

    with pytest.raises(RuntimeError, match=f"Rank 1.*Expected cache path: {cache_path}"):
        StreamingModule(nn.Identity(), 8, tile_cache_path=cache_path)

    assert DummyConstructor.calls == []
