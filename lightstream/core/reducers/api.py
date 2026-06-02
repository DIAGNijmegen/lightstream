"""Reducer authoring base classes and validation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import torch
import torch.nn as nn

from lightstream.core.reducer.base import BaseStreamingGlobalReducer, ReducerMeta, ReducerTile
from lightstream.core.reducer.utils import normalize_spatial_mask, resolve_accumulator_dtype
from lightstream.core.scnn.utils import Box


def validate_arity(inputs: Sequence[torch.Tensor], expected: int | None, *, reducer_name: str) -> None:
    """Validate the number of tensor inputs supplied to a reducer."""
    if expected is None:
        if len(inputs) == 0:
            raise ValueError(f"{reducer_name} expects at least one tensor input.")
        return
    if len(inputs) != expected:
        raise ValueError(f"{reducer_name} expects exactly {expected} tensor input(s), got {len(inputs)}.")


def validate_nchw_shape(tensor: torch.Tensor, *, name: str = "input", reducer_name: str = "Reducer") -> torch.Tensor:
    """Validate that ``tensor`` is an NCHW tensor and return it unchanged."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{reducer_name} {name} must be a torch.Tensor, got {type(tensor)!r}.")
    if tensor.ndim != 4:
        raise ValueError(f"{reducer_name} {name} must be NCHW, got shape={tuple(tensor.shape)}.")
    return tensor


def validate_channel_compatibility(
    tensors: Sequence[torch.Tensor],
    *,
    reducer_name: str = "Reducer",
    allow_single_channel: bool = True,
) -> None:
    """Validate that aligned inputs have compatible channel dimensions.

    By default an input with channel count ``1`` is accepted as broadcastable
    against the reference input's channel count, which matches common spatial
    attention/logit reducers.
    """
    if len(tensors) <= 1:
        return
    reference_channels = int(tensors[0].shape[1])
    allowed = {reference_channels}
    if allow_single_channel:
        allowed.add(1)
    for idx, tensor in enumerate(tensors[1:], start=1):
        channels = int(tensor.shape[1])
        if channels not in allowed:
            raise ValueError(
                f"{reducer_name} input {idx} channel dim must be compatible with C={reference_channels}"
                f"{' or 1' if allow_single_channel else ''}, got {channels}."
            )


def validate_aligned_nchw_inputs(
    inputs: Sequence[torch.Tensor],
    *,
    expected: int | None = None,
    reducer_name: str = "Reducer",
    require_channel_compatibility: bool = True,
    allow_single_channel: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Validate arity plus NCHW batch/spatial alignment for reducer inputs."""
    validate_arity(inputs, expected, reducer_name=reducer_name)
    tensors = tuple(validate_nchw_shape(tensor, name=f"input {idx}", reducer_name=reducer_name) for idx, tensor in enumerate(inputs))
    if not tensors:
        raise ValueError(f"{reducer_name} expects at least one tensor input.")

    batch = int(tensors[0].shape[0])
    spatial = tuple(int(v) for v in tensors[0].shape[-2:])
    for idx, tensor in enumerate(tensors[1:], start=1):
        if int(tensor.shape[0]) != batch or tuple(int(v) for v in tensor.shape[-2:]) != spatial:
            raise ValueError(
                f"{reducer_name} input {idx} must match batch/spatial dimensions "
                f"N={batch}, H/W={spatial}, got shape={tuple(tensor.shape)}."
            )
    if require_channel_compatibility:
        validate_channel_compatibility(tensors, reducer_name=reducer_name, allow_single_channel=allow_single_channel)
    return tensors


def validate_mask_shape(mask: torch.Tensor | None, reference: torch.Tensor, *, reducer_name: str = "Reducer") -> torch.Tensor | None:
    """Validate and normalize an optional spatial mask to ``[N, 1, H, W]``."""
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{reducer_name} mask must be a torch.Tensor, got {type(mask)!r}.")
    try:
        return normalize_spatial_mask(mask, reference)
    except ValueError as exc:
        raise ValueError(f"{reducer_name} mask shape is incompatible with input shape {tuple(reference.shape)}: {exc}") from exc


class BaseReducer(nn.Module, ABC):
    """Base class for reducer modules with offline and streaming forms."""

    def __init__(self, *, streaming_passthrough: bool = False, streaming_forward: bool = False):
        super().__init__()
        self._streaming_passthrough = bool(streaming_passthrough)
        self._streaming_forward = bool(streaming_forward)

    @abstractmethod
    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run reducer logic over one or more NCHW tensors."""

    @abstractmethod
    def to_streaming(self) -> BaseStreamingGlobalReducer:
        """Create the equivalent streaming reducer implementation."""


class SpatialReducer(BaseReducer, ABC):
    """Base class for reducers that consume one NCHW tensor."""

    input_arity = 1

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reduce ``x`` while preserving normal non-streaming behavior by default."""
        x = validate_nchw_shape(x, reducer_name=type(self).__name__)
        normalized_mask = validate_mask_shape(mask, x, reducer_name=type(self).__name__)
        if self._streaming_passthrough:
            return x
        if self._streaming_forward:
            return self._run_streaming_forward((x,), normalized_mask)
        return self.reduce_spatial(x, mask=normalized_mask)

    def _run_streaming_forward(self, inputs: tuple[torch.Tensor, ...], mask: torch.Tensor | None) -> torch.Tensor:
        """Run init_state/update/finalize over a full tensor as one stream tile."""
        x = inputs[0]
        streaming = self.to_streaming()
        acc_dtype = resolve_accumulator_dtype(getattr(streaming, "accumulator_dtype", None), x.dtype)
        meta = ReducerMeta(
            output_height=int(x.shape[-2]),
            output_width=int(x.shape[-1]),
            batch_size=int(x.shape[0]),
            channels=int(x.shape[1]),
            device=x.device,
            dtype=x.dtype,
            accumulator_dtype=acc_dtype,
        )
        state = streaming.init_state(meta)
        tile_mask = None
        if mask is not None:
            # ReducerTile masks are 2D in engine-normalized coordinates. Full
            # forward streaming only supports a shared spatial mask.
            if mask.shape[0] != 1 and mask.shape[0] != x.shape[0]:
                raise ValueError(f"{type(self).__name__} mask batch dim must be 1 or N={x.shape[0]}, got {mask.shape[0]}.")
            if mask.shape[0] != 1 or not torch.all(mask == mask[:1]):
                raise ValueError(f"{type(self).__name__} streaming forward requires a batch-shared spatial mask.")
            tile_mask = mask[0, 0].to(device=x.device, dtype=torch.bool)
        tile = ReducerTile(tensors=inputs, mask=tile_mask, box=Box(0, int(x.shape[-2]), 0, int(x.shape[-1]), None), is_new=tile_mask)
        state = streaming.update(state, tile)
        return streaming.finalize(state)

    @abstractmethod
    def reduce_spatial(self, x: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reduce one validated NCHW tensor over its spatial dimensions."""


class MultiInputSpatialReducer(BaseReducer, ABC):
    """Base class for reducers consuming several aligned NCHW tensors."""

    expected_inputs: int | None = None
    require_channel_compatibility = True
    allow_single_channel = True

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, ...]:
        tensors = self.require_aligned_nchw(inputs)
        normalized_mask = validate_mask_shape(mask, tensors[0], reducer_name=type(self).__name__)
        if self._streaming_passthrough:
            return tensors[0] if len(tensors) == 1 else tensors
        if self._streaming_forward:
            return self._run_streaming_forward(tensors, normalized_mask)
        return self.reduce_spatial_inputs(*tensors, mask=normalized_mask)

    @classmethod
    def require_aligned_nchw(cls, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Validate multi-input NCHW arity, channel compatibility, and alignment."""
        return validate_aligned_nchw_inputs(
            inputs,
            expected=cls.expected_inputs,
            reducer_name=cls.__name__,
            require_channel_compatibility=cls.require_channel_compatibility,
            allow_single_channel=cls.allow_single_channel,
        )

    def _run_streaming_forward(self, inputs: tuple[torch.Tensor, ...], mask: torch.Tensor | None) -> torch.Tensor:
        return SpatialReducer._run_streaming_forward(self, inputs, mask)

    @abstractmethod
    def reduce_spatial_inputs(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reduce validated aligned NCHW tensor inputs."""


class ManualVJPReducer(BaseStreamingGlobalReducer, ABC):
    """Streaming reducer base for reducers with custom backward replay math."""

    @abstractmethod
    def reduce_tile_for_backward(self, tile: ReducerTile, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        """Replay tile-local reduction used by manual VJP implementations."""


ManualBackwardReducer = ManualVJPReducer

__all__ = [
    "BaseReducer",
    "SpatialReducer",
    "MultiInputSpatialReducer",
    "ManualVJPReducer",
    "ManualBackwardReducer",
    "validate_arity",
    "validate_nchw_shape",
    "validate_channel_compatibility",
    "validate_aligned_nchw_inputs",
    "validate_mask_shape",
]
