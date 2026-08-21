"""Capabilities and adapters for operators used by the streaming engine.

This module is deliberately independent of :mod:`scnn`: integrations can inspect
or extend the registry without importing the compatibility facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class OperatorCapabilities:
    """The pieces of the streaming contract implemented by an operator."""

    conversion: bool
    statistics_forward: bool = False
    statistics_backward: bool = False
    spatial_preserving: bool = False
    backward_tile_state: bool = False
    alignment: bool = False


@runtime_checkable
class StreamingOperatorAdapter(Protocol):
    """Protocol implemented by entries in :class:`StreamingOperatorRegistry`."""

    capabilities: OperatorCapabilities

    def to_streaming(self, module: torch.nn.Module, facade: object) -> torch.nn.Module: ...

    def from_streaming(self, module: torch.nn.Module, facade: object) -> torch.nn.Module: ...


@dataclass(frozen=True)
class OperatorAdapter:
    capabilities: OperatorCapabilities
    convert: Callable[[torch.nn.Module, object], torch.nn.Module] = lambda module, facade: module
    restore: Callable[[torch.nn.Module, object], torch.nn.Module] = lambda module, facade: module

    def to_streaming(self, module: torch.nn.Module, facade: object) -> torch.nn.Module:
        return self.convert(module, facade)

    def from_streaming(self, module: torch.nn.Module, facade: object) -> torch.nn.Module:
        return self.restore(module, facade)


class StreamingOperatorRegistry:
    """Type based registry with normal Python MRO/subclass matching."""

    def __init__(self) -> None:
        self._adapters: dict[type[torch.nn.Module], StreamingOperatorAdapter] = {}

    def register(self, module_type: type[torch.nn.Module], adapter: StreamingOperatorAdapter) -> None:
        self._adapters[module_type] = adapter

    def adapter_for(self, module: torch.nn.Module) -> StreamingOperatorAdapter | None:
        for cls in type(module).__mro__:
            if cls in self._adapters:
                return self._adapters[cls]
        return None

    def capabilities_for(self, module: torch.nn.Module) -> OperatorCapabilities | None:
        adapter = self.adapter_for(module)
        return adapter.capabilities if adapter is not None else None


STREAMING_OPERATORS = StreamingOperatorRegistry()


def register_operator(module_type: type[torch.nn.Module], **kwargs: object) -> None:
    """Convenience API for third-party operator integrations."""
    convert = kwargs.pop("convert", lambda module, facade: module)
    restore = kwargs.pop("restore", lambda module, facade: module)
    STREAMING_OPERATORS.register(
        module_type,
        OperatorAdapter(OperatorCapabilities(**kwargs), convert=convert, restore=restore),  # type: ignore[arg-type]
    )


def _register_builtin_operators() -> None:
    # Geometry-bearing PyTorch operators participate in statistics collection.
    register_operator(torch.nn.Conv2d, conversion=True, statistics_forward=True, statistics_backward=True, alignment=True)
    register_operator(torch.nn.MaxPool2d, conversion=True, statistics_forward=True, statistics_backward=True, alignment=True)
    register_operator(torch.nn.AvgPool2d, conversion=True, statistics_forward=True, alignment=True)
    register_operator(torch.nn.Upsample, conversion=True, statistics_forward=True, statistics_backward=True)

    # These eager modules do not alter spatial support and need no conversion.
    pointwise = (
        torch.nn.Identity, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.GELU,
        torch.nn.SiLU, torch.nn.ELU, torch.nn.LeakyReLU, torch.nn.Dropout,
        torch.nn.Dropout2d, torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d,
    )
    for module_type in pointwise:
        register_operator(module_type, conversion=True, spatial_preserving=True)


_register_builtin_operators()
