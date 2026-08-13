__all__ = [
    "LightningStreamingModule",
    "ImageNetClassifier",
]


def __getattr__(name):
    """Load Lightning-dependent convenience classes only when requested."""
    if name == "LightningStreamingModule":
        from .lightningstreaming import LightningStreamingModule

        return LightningStreamingModule
    if name == "ImageNetClassifier":
        from .imagenet_template import ImageNetClassifier

        return ImageNetClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
