from __future__ import annotations

__all__ = ["StreamingWSS"]

def __getattr__(name: str):
    if name == "StreamingWSS":
        from .streamingwss import StreamingWSS
        return StreamingWSS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")