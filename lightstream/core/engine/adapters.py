"""Layer-adapter extension points for streaming conversion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch.nn as nn

ModuleT = TypeVar("ModuleT", bound=nn.Module)


@runtime_checkable
class StreamableLayerAdapter(Protocol):
    """Protocol for user-defined layer conversion adapters.

    Adapters are intentionally small operator-level extension points.  They
    describe how one layer type is translated to and from an equivalent
    streaming implementation, while the streaming engine remains responsible for
    orchestration through composition.
    """

    def can_adapt(self, module: nn.Module) -> bool:
        """Return ``True`` when this adapter can convert ``module``."""

    def to_streaming(self, module: nn.Module, *, context: dict[str, Any] | None = None) -> nn.Module:
        """Convert a regular layer to its streaming equivalent."""

    def from_streaming(self, module: nn.Module, *, context: dict[str, Any] | None = None) -> nn.Module:
        """Convert a streaming layer back to a regular PyTorch layer."""


@dataclass
class AdapterRegistry:
    """Ordered registry of streamable layer adapters.

    The registry is owned by :class:`~lightstream.core.engine.api.StreamingEngine`
    so future conversion paths can be configured without introducing a large
    engine inheritance hierarchy.  The first adapter whose ``can_adapt`` method
    accepts a module is selected.
    """

    adapters: list[StreamableLayerAdapter] = field(default_factory=list)

    def register(self, adapter: StreamableLayerAdapter) -> StreamableLayerAdapter:
        """Register ``adapter`` and return it for decorator-style usage."""
        if not isinstance(adapter, StreamableLayerAdapter):
            raise TypeError("adapter must implement StreamableLayerAdapter")
        self.adapters.append(adapter)
        return adapter

    def find(self, module: nn.Module) -> StreamableLayerAdapter | None:
        """Find the first adapter that can convert ``module``."""
        for adapter in self.adapters:
            if adapter.can_adapt(module):
                return adapter
        return None
