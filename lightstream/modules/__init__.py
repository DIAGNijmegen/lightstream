from __future__ import annotations

__all__ = [
    "LightningStreamingModule",
    "ImageNetClassifier",
    "GlobalLossReducer",
    "StreamingGlobalLossReducer",
    "GlobalWSLossReducer",
    "StreamingGlobalWSLossReducer",
]


def __getattr__(name: str):
    if name == "LightningStreamingModule":
        from .lightningstreaming import LightningStreamingModule

        return LightningStreamingModule
    if name == "ImageNetClassifier":
        from .imagenet_template import ImageNetClassifier

        return ImageNetClassifier
    if name in {"GlobalLossReducer", "GlobalWSLossReducer"}:
        from .loss_reducer import GlobalLossReducer

        return GlobalLossReducer
    if name in {"StreamingGlobalLossReducer", "StreamingGlobalWSLossReducer"}:
        from .loss_reducer import StreamingGlobalLossReducer

        return StreamingGlobalLossReducer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
