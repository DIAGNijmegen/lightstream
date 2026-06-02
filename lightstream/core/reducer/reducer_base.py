"""Abstract base types for reducer extension points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

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


class SpatialReducer(BaseReducer, ABC):
    """Base class for single-input spatial reducers.

    Subclasses implement :meth:`reduce_spatial` for the math/operator behavior.
    The engine can continue to treat reducers uniformly through ``BaseReducer``
    while users get a focused inheritance point for custom NCHW spatial
    reductions.
    """

    input_arity = 1

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.require_single_nchw(inputs)
        if self._streaming_passthrough:
            return x
        return self.reduce_spatial(x, mask=mask)

    @staticmethod
    def require_single_nchw(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Validate and return the sole NCHW tensor input."""
        if len(inputs) != 1:
            raise ValueError(f"SpatialReducer expects exactly one tensor input, got {len(inputs)}.")
        x = inputs[0]
        if x.ndim != 4:
            raise ValueError(f"SpatialReducer expects an NCHW tensor, got shape={tuple(x.shape)}")
        return x

    def reduce_spatial(self, x: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reduce one NCHW tensor over its spatial dimensions."""
        raise NotImplementedError(f"{type(self).__name__}.reduce_spatial() must be implemented")


class MultiInputSpatialReducer(BaseReducer, ABC):
    """Base class for reducers that consume multiple aligned spatial tensors.

    ``expected_inputs`` may be set by subclasses to enforce arity.  Inputs are
    validated as NCHW tensors with matching batch and spatial dimensions before
    :meth:`reduce_spatial_inputs` is called.
    """

    expected_inputs: int | None = None

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        tensors = self.require_aligned_nchw(inputs)
        if self._streaming_passthrough:
            return tensors[0]
        return self.reduce_spatial_inputs(*tensors, mask=mask)

    @classmethod
    def require_aligned_nchw(cls, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Validate multi-input NCHW arity and spatial alignment."""
        if cls.expected_inputs is not None and len(inputs) != cls.expected_inputs:
            raise ValueError(f"{cls.__name__} expects exactly {cls.expected_inputs} tensor inputs, got {len(inputs)}.")
        if len(inputs) == 0:
            raise ValueError(f"{cls.__name__} expects at least one tensor input.")
        tensors = tuple(inputs)
        first = tensors[0]
        if first.ndim != 4:
            raise ValueError(f"{cls.__name__} expects NCHW tensors, got shape={tuple(first.shape)}")
        batch = first.shape[0]
        spatial = first.shape[-2:]
        for idx, tensor in enumerate(tensors[1:], start=1):
            if tensor.ndim != 4:
                raise ValueError(f"{cls.__name__} input {idx} must be NCHW, got shape={tuple(tensor.shape)}")
            if tensor.shape[0] != batch or tensor.shape[-2:] != spatial:
                raise ValueError(
                    f"{cls.__name__} input {idx} must match batch/spatial dimensions "
                    f"N={batch}, H/W={tuple(spatial)}, got shape={tuple(tensor.shape)}"
                )
        return tensors

    def reduce_spatial_inputs(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reduce aligned NCHW tensor inputs over their spatial dimensions."""
        raise NotImplementedError(f"{type(self).__name__}.reduce_spatial_inputs() must be implemented")


class ManualVJPReducer(BaseStreamingGlobalReducer, ABC):
    """Streaming reducer base for implementations with manual VJP replay.

    Custom streaming reducers can inherit from this class when they provide a
    reducer-specific :meth:`reduce_tile_for_backward` expression instead of
    relying on a generic autograd reduction formula.
    """


ManualBackwardReducer = ManualVJPReducer
