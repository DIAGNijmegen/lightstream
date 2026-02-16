from __future__ import annotations

__all__ = ["LightningStreamingModule"]


def __getattr__(name: str):
    if name == "LightningStreamingModule":
        from .modules.lightningstreaming import LightningStreamingModule

        return LightningStreamingModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
