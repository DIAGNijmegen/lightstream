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
    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run non-streaming reduction over positional ``inputs``.

        Conventions
        -----------
        - Legacy reducers consume exactly one tensor input (``inputs[0]``).
        - Multi-input reducers must define and validate expected arity and
          per-input shape contracts.
        - ``mask`` is always a keyword-only auxiliary argument and is not part
          of ``*inputs``.
        """

    @abstractmethod
    def to_streaming(self) -> BaseStreamingGlobalReducer:
        """Create the equivalent streaming reducer implementation."""
