"""Streaming reducer runtime primitives.

The public streaming reducer contract is intentionally small:

``init_state(meta) -> state``
``update(state, ReducerTile) -> state``
``finalize(state) -> tensor``

The engine-facing adapter in this module owns tile geometry, overlap-safe
accounting, reducer mask slicing, replay metadata, and output placement hooks so
reducer authors only implement reduction math.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from lightstream.core.scnn.utils import Box

from .utils import resolve_accumulator_dtype


@dataclass(frozen=True)
class ReducerMeta:
    """Metadata supplied by the engine when a reducer stream starts."""

    output_height: int
    output_width: int
    batch_size: int
    channels: int
    device: torch.device
    dtype: torch.dtype
    accumulator_dtype: torch.dtype


@dataclass(frozen=True)
class ReducerTile:
    """Engine-normalized tile payload passed to user reducers.

    Attributes
    ----------
    tensors:
        One or more aligned NCHW tile tensors. Multi-input reducers receive a
        tuple whose tensors share batch and spatial dimensions.
    mask:
        Effective 2D bool mask in reducer-output coordinates after overlap
        accounting and optional user mask slicing. ``True`` pixels contribute.
    box:
        Tile placement in reducer-output coordinates.
    is_new:
        2D bool mask identifying pixels that have not been seen by earlier
        overlapping tiles, before applying any user mask.
    """

    tensors: tuple[torch.Tensor, ...]
    mask: torch.Tensor | None
    box: Box
    is_new: torch.Tensor | None


@dataclass(frozen=True)
class ReducerReplayRecord:
    """Engine-owned metadata used to validate reducer backward replay."""

    tile_y: int
    tile_x: int
    top: bool
    left: bool
    right: bool
    bottom: bool
    height: int
    width: int
    dst_y0: int
    dst_y1: int
    dst_x0: int
    dst_x1: int
    arity: int


class StreamingReducerTileF(torch.autograd.Function):
    """Tile-local spatial reducer autograd primitive."""

    @staticmethod
    def forward(ctx, tile_output: torch.Tensor, valid_mask: torch.Tensor | None, normalization: torch.Tensor | None) -> torch.Tensor:
        if tile_output.ndim != 4:
            raise ValueError(f"StreamingReducer expects NCHW tile, got shape={tuple(tile_output.shape)}")

        if valid_mask is not None:
            if valid_mask.ndim != 2:
                raise ValueError(f"valid_mask must be 2D (H,W), got shape={tuple(valid_mask.shape)}")
            mask_4d = valid_mask.to(dtype=tile_output.dtype, device=tile_output.device)[None, None]
            masked = tile_output * mask_4d
            ctx.save_for_backward(mask_4d)
            ctx.has_mask = True
        else:
            masked = tile_output
            ctx.save_for_backward(torch.zeros(0, device=tile_output.device, dtype=tile_output.dtype))
            ctx.has_mask = False

        ctx.input_height = tile_output.shape[-2]
        ctx.input_width = tile_output.shape[-1]

        if normalization is not None:
            norm = normalization.to(device=tile_output.device)
            acc_dtype = resolve_accumulator_dtype(norm.dtype, tile_output.dtype)
            norm = norm.to(dtype=acc_dtype).clamp_min(1)
            reduced_acc = masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype) / norm
            reduced = reduced_acc.to(dtype=tile_output.dtype)
            ctx.normalization = norm
            ctx.has_normalization = True
        else:
            acc_dtype = resolve_accumulator_dtype(None, tile_output.dtype)
            reduced = masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=tile_output.dtype)
            ctx.normalization = None
            ctx.has_normalization = False
        return reduced

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mask_4d,) = ctx.saved_tensors
        grad_input = grad_output
        if ctx.has_normalization:
            grad_input = (grad_input.to(dtype=ctx.normalization.dtype) / ctx.normalization).to(dtype=grad_output.dtype)
        grad_input = grad_input.expand(-1, -1, ctx.input_height, ctx.input_width)
        if ctx.has_mask:
            grad_input = grad_input * mask_4d.to(dtype=grad_input.dtype, device=grad_input.device)
        return grad_input, None, None


streaming_reduce_tile = StreamingReducerTileF.apply


class BaseStreamingGlobalReducer(nn.Module, ABC):
    """Engine adapter for the reducer ``init_state/update/finalize`` API.

    Subclasses implement the user-facing reducer lifecycle. This base class is
    responsible for legacy engine integration, including overlap-safe tile
    accounting, replay metadata, and passthrough bookkeeping used by the engine
    to discover reducer heads and aligned multi-input payloads.
    """

    def __init__(self, mode: str = "mean", accumulator_dtype: torch.dtype | None = None):
        super().__init__()
        self.mode = mode
        self.accumulator_dtype = accumulator_dtype
        self._streaming_passthrough = False
        self.register_buffer("running_sum", torch.zeros(0), persistent=False)
        self.register_buffer("running_count", torch.zeros(0), persistent=False)
        self.register_buffer("_stream_seen_mask", torch.zeros(0, dtype=torch.bool), persistent=False)
        self._state: Any = None
        self._passthrough_inputs: tuple[torch.Tensor, ...] | None = None
        self._passthrough_output: torch.Tensor | None = None
        self._debug_replay_enabled = False
        self._replay_assignments: list[ReducerReplayRecord] | None = None
        self._replay_cursor: int | None = None

    @staticmethod
    def _parse_multi_input_payload(payload: torch.Tensor | Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if isinstance(payload, torch.Tensor):
            return (payload,)
        if not isinstance(payload, (tuple, list)):
            raise TypeError(f"Expected tile payload to be tensor/tuple/list, got {type(payload)!r}.")
        if len(payload) == 0:
            raise ValueError("Structured tile payload must contain at least one tensor.")
        if not all(isinstance(t, torch.Tensor) for t in payload):
            bad_type = next(type(t) for t in payload if not isinstance(t, torch.Tensor))
            raise TypeError(f"Structured tile payload elements must be tensors; got {bad_type!r}.")
        return tuple(payload)

    @staticmethod
    def _parse_single_input_payload(payload: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        tensors = BaseStreamingGlobalReducer._parse_multi_input_payload(payload)
        if len(tensors) != 1:
            raise ValueError(f"Single-input reducer expected arity=1, got {len(tensors)}.")
        return tensors[0]

    def reset_stream_state(self, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, accumulator_dtype: torch.dtype | None = None):
        resolved_acc_dtype = resolve_accumulator_dtype(accumulator_dtype or self.accumulator_dtype, dtype)
        self.running_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=dtype)
        self.running_count = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=resolved_acc_dtype)

    def start_stream(self, output_height: int, output_width: int, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, debug_replay: bool = False):
        acc_dtype = resolve_accumulator_dtype(self.accumulator_dtype, dtype)
        self.reset_stream_state(batch_size=batch_size, channels=channels, device=device, dtype=dtype, accumulator_dtype=acc_dtype)
        self._stream_seen_mask = torch.zeros((output_height, output_width), dtype=torch.bool, device=device)
        self._debug_replay_enabled = debug_replay
        self._replay_assignments = [] if debug_replay else None
        self._replay_cursor = None
        self._state = self.init_state(
            ReducerMeta(
                output_height=int(output_height),
                output_width=int(output_width),
                batch_size=int(batch_size),
                channels=int(channels),
                device=device,
                dtype=dtype,
                accumulator_dtype=acc_dtype,
            )
        )

    def accumulate_stream_tile(self, trimmed_output: torch.Tensor | Sequence[torch.Tensor], tile_y: int, tile_x: int, sides, dst_box, user_mask: torch.Tensor | None = None):
        tensors = self._parse_multi_input_payload(trimmed_output)
        ref = tensors[0]
        dst_y0, dst_y1, dst_x0, dst_x1 = (int(v) for v in dst_box)
        seen_slice = self._stream_seen_mask[dst_y0:dst_y1, dst_x0:dst_x1]
        is_new = ~seen_slice
        effective_mask = is_new if user_mask is None else (is_new & user_mask.to(dtype=torch.bool, device=is_new.device))
        box = Box(dst_y0, -1, dst_x0, -1, sides)

        if self._debug_replay_enabled:
            if self._replay_assignments is None:
                raise RuntimeError("Reducer replay assignments are not initialized.")
            self._replay_assignments.append(
                ReducerReplayRecord(
                    tile_y=int(tile_y),
                    tile_x=int(tile_x),
                    top=bool(sides.top),
                    left=bool(sides.left),
                    right=bool(sides.right),
                    bottom=bool(sides.bottom),
                    height=int(ref.shape[-2]),
                    width=int(ref.shape[-1]),
                    dst_y0=dst_y0,
                    dst_y1=dst_y1,
                    dst_x0=dst_x0,
                    dst_x1=dst_x1,
                    arity=len(tensors),
                )
            )

        if torch.any(effective_mask):
            self._state = self.update(self._state, ReducerTile(tensors=tensors, mask=effective_mask, box=box, is_new=is_new))
        seen_slice |= is_new

    def finish_stream(self) -> torch.Tensor:
        return self.finalize_stream()

    def finalize_stream(self) -> torch.Tensor:
        if self._state is None:
            raise RuntimeError("Reducer stream state is empty. Call start_stream() first.")
        return self.finalize(self._state)

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if len(inputs) == 0:
            raise ValueError(f"{type(self).__name__} passthrough expects at least one tensor input.")
        self._passthrough_inputs = tuple(inputs)
        self._passthrough_output = inputs[0]
        return inputs[0] if len(inputs) == 1 else tuple(inputs)

    def start_backward_replay(self):
        if self._debug_replay_enabled:
            if self._replay_assignments is None:
                raise RuntimeError("Reducer replay assignments are not available for backward replay.")
            self._replay_cursor = 0
        else:
            self._replay_cursor = None

    def validate_backward_replay_consumed(self, *, head_idx: int):
        if not self._debug_replay_enabled:
            return
        if self._replay_assignments is None or self._replay_cursor is None:
            raise RuntimeError("Reducer replay state is not initialized.")
        if self._replay_cursor != len(self._replay_assignments):
            raise RuntimeError(f"Reducer assignment replay incomplete for head {head_idx}: consumed={self._replay_cursor}, expected={len(self._replay_assignments)}")

    def build_backward_pair(self, trimmed_output: torch.Tensor | Sequence[torch.Tensor], gradient: torch.Tensor, *, input_y: int, input_x: int, sides, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self._parse_multi_input_payload(trimmed_output)
        ref = tensors[0]
        if self._debug_replay_enabled:
            if self._replay_assignments is None or self._replay_cursor is None:
                raise RuntimeError("Reducer replay state is not initialized. Call start_backward_replay() first.")
            self._replay_cursor = self._validate_replay_assignment(
                assignments=self._replay_assignments,
                cursor=self._replay_cursor,
                input_y=input_y,
                input_x=input_x,
                sides=sides,
                expected_h=int(ref.shape[-2]),
                expected_w=int(ref.shape[-1]),
                expected_arity=len(tensors),
            )
        box = Box(int(input_y), -1, int(input_x), -1, sides)
        tile = ReducerTile(tensors=tensors, mask=valid_mask, box=box, is_new=valid_mask)
        reduced_output = self.reduce_tile_for_backward(tile, global_context=self.extra_state_for_backward())
        return reduced_output, gradient

    @abstractmethod
    def init_state(self, meta: ReducerMeta) -> Any:
        """Initialize and return reducer state for a stream."""

    @abstractmethod
    def update(self, state: Any, tile: ReducerTile) -> Any:
        """Fold one engine-normalized tile into reducer state."""

    @abstractmethod
    def finalize(self, state: Any) -> torch.Tensor:
        """Return the finalized reducer output."""

    def reduce_tile_for_backward(self, tile: ReducerTile, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        """Advanced escape hatch for custom reducer backward replay.

        Reducers with non-trivial VJPs can override this method. The default is
        correct for sum-style single-input reducers.
        """
        normalization = global_context.get("normalization") if global_context else None
        return streaming_reduce_tile(tile.tensors[0], tile.mask, normalization)

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        return {}

    def _validate_replay_assignment(self, *, assignments: list[ReducerReplayRecord], cursor: int, input_y: int, input_x: int, sides, expected_h: int, expected_w: int, expected_arity: int) -> int:
        if cursor >= len(assignments):
            raise RuntimeError("Reducer assignment cursor out of range.")
        record = assignments[cursor]
        if (
            int(input_y) != record.tile_y
            or int(input_x) != record.tile_x
            or bool(sides.top) != record.top
            or bool(sides.left) != record.left
            or bool(sides.right) != record.right
            or bool(sides.bottom) != record.bottom
        ):
            raise RuntimeError(
                "Reducer tile replay mismatch: "
                f"forward tile=({record.tile_y},{record.tile_x},{record.top},{record.left},{record.right},{record.bottom}) "
                f"backward tile=({int(input_y)},{int(input_x)},{bool(sides.top)},{bool(sides.left)},{bool(sides.right)},{bool(sides.bottom)})"
            )
        if expected_h != record.height or expected_w != record.width:
            raise RuntimeError(f"Reducer trimmed shape mismatch: forward=({record.height},{record.width}) backward=({expected_h},{expected_w})")
        if (record.dst_y1 - record.dst_y0) != expected_h or (record.dst_x1 - record.dst_x0) != expected_w:
            raise RuntimeError(
                "Reducer assignment mismatch: "
                f"stored=({record.dst_y0}:{record.dst_y1},{record.dst_x0}:{record.dst_x1}) current=({expected_h},{expected_w})"
            )
        if record.arity != int(expected_arity):
            raise RuntimeError(f"Reducer input arity mismatch: forward={record.arity} backward={int(expected_arity)}")
        return cursor + 1


class StreamingReducer(BaseStreamingGlobalReducer, ABC):
    """Backward-compatible abstract alias for custom streaming reducers."""
