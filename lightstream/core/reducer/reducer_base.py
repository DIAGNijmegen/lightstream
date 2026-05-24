"""Abstract base types for non-streaming global reducers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .base import BaseStreamingGlobalReducer


class BaseReducer(nn.Module, ABC):
    """Base class for non-streaming reducers.

    Non-streaming reducers implement offline/global reduction in :meth:`forward`
    and must provide deterministic conversion to a streaming reducer via
    :meth:`to_streaming` with equivalent reduction semantics.

    Parameters
    ----------
    streaming_passthrough : bool, default=False
        When ``True``, :meth:`forward` must return the input unchanged. This is
        used for reducer-head tagging in SCNN where execution is deferred to
        streaming orchestration.
    """

    def __init__(self, *, streaming_passthrough: bool = False):
        super().__init__()
        self._streaming_passthrough = bool(streaming_passthrough)

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run non-streaming reduction on ``x`` (or passthrough when enabled)."""

    @abstractmethod
    def to_streaming(self) -> BaseStreamingGlobalReducer:
        """Create the equivalent streaming reducer implementation."""

