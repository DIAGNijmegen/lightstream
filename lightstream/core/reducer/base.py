import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Sequence

from .utils import resolve_accumulator_dtype


class StreamingReducerTileF(torch.autograd.Function):
    """Tile-local spatial reducer autograd primitive.

    This function computes a spatial sum (or normalized sum) for a single tile and
    returns a tensor of shape ``[N, C, 1, 1]``. In backward, gradients are expanded
    back to tile shape and masked when a valid mask is provided.
    """

    @staticmethod
    def forward(
        ctx,
        tile_output: torch.Tensor,
        valid_mask: torch.Tensor | None,
        normalization: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reduce one tile contribution.

        Parameters
        ----------
        tile_output : torch.Tensor
            Tile tensor with shape ``[N, C, H, W]``.
        valid_mask : torch.Tensor | None
            Optional spatial mask with shape ``[H, W]`` that marks valid pixels.
        normalization : torch.Tensor | None
            Optional divisor tensor, usually running counts for mean mode.

        Returns
        -------
        torch.Tensor
            Reduced tile output with shape ``[N, C, 1, 1]``.
        """
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
        """Propagate gradients from reduced output back to tile layout.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient at reduced output with shape ``[N, C, 1, 1]``.

        Returns
        -------
        tuple[torch.Tensor, None, None]
            Gradient for tile input and ``None`` for non-differentiable inputs.
        """
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
    """Abstract base class for streaming global reducers.

    This base class owns:
    - stream lifecycle setup/reset (`start_stream`, `reset_stream_state`)
    - overlap-safe tile orchestration (`accumulate_stream_tile`)
    - stream finalization (`finish_stream`/`finalize_stream`)
    - debug backward replay bookkeeping (`start_backward_replay`,
      `build_backward_pair`, replay validation).

    Subclasses must implement:
    - `init_reduction_state(...)`
    - `accumulate_valid_tile(tile, valid_mask)`
    - `finalize_from_state()`
    - `reduce_tile_for_backward(trimmed_output, valid_mask, global_context)`

    Subclasses may override:
    - `extra_state_for_backward()` to expose global tensors/scalars needed during
      backward replay tile reduction.

    Base guarantees:
    - non-overlapping accounting across streamed tiles via `_stream_seen_mask`
      semantics (parity with previous mean reducer behavior)
    - centralized accumulator dtype policy helpers (minimum fp32 when unresolved
      by caller/subclass through `resolve_accumulator_dtype`)
    - consistent replay metadata validation between forward tile traversal and
      backward replay traversal.

    This class owns stream lifecycle, tile accumulation state, optional replay
    bookkeeping, and final reduction output assembly.

    Parameters
    ----------
    mode : str, default="mean"
        Reduction mode. Must be ``"sum"`` or ``"mean"``.
    accumulator_dtype : torch.dtype | None, default=None
        Optional accumulator dtype for numerically stable reduction.
    """

    @staticmethod
    def _parse_single_input_payload(payload: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """Normalize reducer payload for legacy single-input reducers."""
        if isinstance(payload, torch.Tensor):
            return payload
        if isinstance(payload, (tuple, list)):
            if len(payload) != 1:
                raise ValueError(
                    "Legacy reducers expect a single tile tensor payload; "
                    f"got structured payload with arity={len(payload)}."
                )
            tile = payload[0]
            if not isinstance(tile, torch.Tensor):
                raise TypeError(f"Expected tensor payload element, got {type(tile)!r}.")
            return tile
        raise TypeError(f"Expected tile payload to be tensor/tuple/list, got {type(payload)!r}.")

    @staticmethod
    def _parse_multi_input_payload(payload: torch.Tensor | Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Normalize reducer payload into a tuple of tile tensors."""
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

    def __init__(self, mode: str = "mean", accumulator_dtype: torch.dtype | None = None):
        super().__init__()
        if mode not in {"sum", "mean"}:
            raise ValueError(f"Unsupported reducer mode '{mode}', expected 'sum' or 'mean'.")
        self.mode = mode
        self.accumulator_dtype = accumulator_dtype
        self._streaming_passthrough = False
        self.register_buffer("running_sum", torch.zeros(0), persistent=False)
        self.register_buffer("running_count", torch.zeros(0), persistent=False)
        self.register_buffer("_stream_seen_mask", torch.zeros(0, dtype=torch.bool), persistent=False)
        self._last_output = None
        self._debug_replay_enabled = False
        self._replay_assignments: list[tuple] | None = None
        self._replay_cursor: int | None = None

    def reset_stream_state(
        self,
        batch_size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
        accumulator_dtype: torch.dtype | None = None,
    ):
        """Reset running reduction buffers.

        Parameters
        ----------
        batch_size : int
            Batch size ``N``.
        channels : int
            Channel count ``C``.
        device : torch.device
            Target buffer device.
        dtype : torch.dtype
            Output accumulation dtype for ``running_sum``.
        accumulator_dtype : torch.dtype | None, default=None
            Optional dtype override for ``running_count``.
        """
        self.running_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=dtype)
        resolved_acc_dtype = resolve_accumulator_dtype(accumulator_dtype or self.accumulator_dtype, dtype)
        self.running_count = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=resolved_acc_dtype)
        self.init_reduction_state(batch_size=batch_size, channels=channels, device=device, dtype=dtype, accumulator_dtype=resolved_acc_dtype)

    def start_stream(
        self,
        output_height: int,
        output_width: int,
        batch_size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
        debug_replay: bool = False,
    ):
        """Initialize reducer state before tile traversal."""
        self.reset_stream_state(batch_size=batch_size, channels=channels, device=device, dtype=dtype)
        self._stream_seen_mask = torch.zeros((output_height, output_width), dtype=torch.bool, device=device)
        self._debug_replay_enabled = debug_replay
        self._replay_assignments = [] if debug_replay else None
        self._replay_cursor = None

    def accumulate_stream_tile(self, trimmed_output: torch.Tensor | Sequence[torch.Tensor], tile_y: int, tile_x: int, sides, dst_box, user_mask: torch.Tensor | None = None):
        """Accumulate one tile while enforcing non-overlapping pixel counting.

        Parameters
        ----------
        trimmed_output : torch.Tensor | Sequence[torch.Tensor]
            Tile payload. A single tensor uses the legacy single-input path.
            Tuple/list payloads represent structured multi-input tiles.
        """
        tile_payload = self._parse_single_input_payload(trimmed_output)
        dst_y0, dst_y1, dst_x0, dst_x1 = dst_box
        seen_slice = self._stream_seen_mask[dst_y0:dst_y1, dst_x0:dst_x1]
        new_mask = ~seen_slice
        effective_mask = new_mask if user_mask is None else (new_mask & user_mask.to(dtype=torch.bool, device=new_mask.device))
        if self._debug_replay_enabled:
            if self._replay_assignments is None:
                raise RuntimeError("Reducer replay assignments are not initialized.")
            self._replay_assignments.append(
                (
                    int(tile_y),
                    int(tile_x),
                    bool(sides.top),
                    bool(sides.left),
                    bool(sides.right),
                    bool(sides.bottom),
                    int(tile_payload.shape[-2]),
                    int(tile_payload.shape[-1]),
                    int(dst_y0),
                    int(dst_y1),
                    int(dst_x0),
                    int(dst_x1),
                    1,
                )
            )
        if torch.any(effective_mask):
            self.accumulate_valid_tile(tile_payload, valid_mask=effective_mask)
        seen_slice |= new_mask

    def finish_stream(self) -> torch.Tensor:
        """Return finalized reduced output for current stream."""
        return self.finalize_stream()

    def start_backward_replay(self):
        """Prepare replay cursor used by debug backward checks."""
        if self._debug_replay_enabled:
            if self._replay_assignments is None:
                raise RuntimeError("Reducer replay assignments are not available for backward replay.")
            self._replay_cursor = 0
        else:
            self._replay_cursor = None

    def validate_backward_replay_consumed(self, *, head_idx: int):
        """Validate that backward replay consumed all recorded assignments."""
        if not self._debug_replay_enabled:
            return
        if self._replay_assignments is None or self._replay_cursor is None:
            raise RuntimeError("Reducer replay state is not initialized.")
        if self._replay_cursor != len(self._replay_assignments):
            raise RuntimeError(f"Reducer assignment replay incomplete for head {head_idx}: consumed={self._replay_cursor}, expected={len(self._replay_assignments)}")

    def finalize_stream(self) -> torch.Tensor:
        """Compute final output from reducer state."""
        return self.finalize_from_state()

    def build_backward_pair(self, trimmed_output: torch.Tensor | Sequence[torch.Tensor], gradient: torch.Tensor, *, input_y: int, input_x: int, sides, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Build reducer output / upstream gradient pair for backward orchestration.

        Parameters
        ----------
        trimmed_output : torch.Tensor | Sequence[torch.Tensor]
            Tile payload for backward replay. Accepts legacy single-input tensor
            or structured tuple/list for multi-input reducers.
        valid_mask : torch.Tensor | None, default=None
            Overlap-safe validity mask supplied by ``StreamingCNN`` for this
            replay tile. Reducers consume it directly and do not own separate
            backward mask replay state.
        """
        tile_payload = self._parse_single_input_payload(trimmed_output)
        expected_h = int(tile_payload.shape[-2])
        expected_w = int(tile_payload.shape[-1])
        if self._debug_replay_enabled:
            if self._replay_assignments is None or self._replay_cursor is None:
                raise RuntimeError("Reducer replay state is not initialized. Call start_backward_replay() first.")
            self._replay_cursor = self._validate_replay_assignment(
                assignments=self._replay_assignments,
                cursor=self._replay_cursor,
                input_y=input_y,
                input_x=input_x,
                sides=sides,
                expected_h=expected_h,
                expected_w=expected_w,
                expected_arity=1,
            )
        global_context = self.extra_state_for_backward()
        reduced_output = self.reduce_tile_for_backward(tile_payload, valid_mask=valid_mask, global_context=global_context)
        return reduced_output, gradient

    @abstractmethod
    def init_reduction_state(self, *, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype, accumulator_dtype: torch.dtype) -> None:
        """Initialize subclass-owned state for a stream."""

    @abstractmethod
    def accumulate_valid_tile(self, tile: torch.Tensor | Sequence[torch.Tensor], valid_mask: torch.Tensor) -> None:
        """Accumulate one valid tile contribution into subclass state.

        ``tile`` may be a single tensor or structured tuple/list payload.
        """

    @abstractmethod
    def finalize_from_state(self) -> torch.Tensor:
        """Finalize and return stream output from subclass state."""

    @abstractmethod
    def reduce_tile_for_backward(self, trimmed_output: torch.Tensor | Sequence[torch.Tensor], valid_mask: torch.Tensor | None, global_context: dict[str, torch.Tensor | int | float | None]) -> torch.Tensor:
        """Build a tile-local replay expression for reducer backward.

        ``trimmed_output`` may be a single tensor or structured tuple/list payload.

        The returned tensor is consumed only by backward replay and is allowed to be
        a derivative surrogate. It does not need to numerically equal the tile's
        final forward contribution, and for nonlinear global reducers such as GeM
        it generally should not finalize each tile independently. Instead, the
        implementation contract is gradient equivalence: when the same upstream
        reducer gradient is applied to every replay tile and those per-tile input
        gradients are summed over all tiles, the result must match the gradient of
        the finalized global reducer output with respect to the original full
        input(s).

        Avoid applying nonlinear finalization (for example, ``pow(1.0 / r)``) to
        each tile contribution unless the math has been derived to be equivalent
        under this gradient-summing contract. Use ``global_context`` for finalized
        global state needed to build an equivalent replay expression.
        """

    def extra_state_for_backward(self) -> dict[str, torch.Tensor | int | float | None]:
        """Optional global state to feed `reduce_tile_for_backward`."""
        return {}

    def _validate_replay_assignment(
        self,
        *,
        assignments: list[tuple],
        cursor: int,
        input_y: int,
        input_x: int,
        sides,
        expected_h: int,
        expected_w: int,
        expected_arity: int,
    ) -> int:
        """Check backward tile metadata against recorded forward metadata."""
        if cursor >= len(assignments):
            raise RuntimeError("Reducer assignment cursor out of range.")
        (f_tile_y, f_tile_x, f_top, f_left, f_right, f_bottom, f_h, f_w, dst_y0, dst_y1, dst_x0, dst_x1, f_arity) = assignments[cursor]
        if (int(input_y) != int(f_tile_y) or int(input_x) != int(f_tile_x) or bool(sides.top) != bool(f_top) or bool(sides.left) != bool(f_left) or bool(sides.right) != bool(f_right) or bool(sides.bottom) != bool(f_bottom)):
            raise RuntimeError("Reducer tile replay mismatch: " f"forward tile=({f_tile_y},{f_tile_x},{f_top},{f_left},{f_right},{f_bottom}) " f"backward tile=({int(input_y)},{int(input_x)},{bool(sides.top)},{bool(sides.left)},{bool(sides.right)},{bool(sides.bottom)})")
        if expected_h != int(f_h) or expected_w != int(f_w):
            raise RuntimeError("Reducer trimmed shape mismatch: " f"forward=({f_h},{f_w}) backward=({expected_h},{expected_w})")
        if (dst_y1 - dst_y0) != expected_h or (dst_x1 - dst_x0) != expected_w:
            raise RuntimeError("Reducer assignment mismatch: " f"stored=({dst_y0}:{dst_y1},{dst_x0}:{dst_x1}) current=({expected_h},{expected_w})")
        if int(f_arity) != int(expected_arity):
            raise RuntimeError(f"Reducer input arity mismatch: forward={int(f_arity)} backward={int(expected_arity)}")
        return cursor + 1

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Streaming forward passthrough for reducer tile payloads.

        Parameters
        ----------
        *inputs : torch.Tensor
            Positional tile payload.
            Legacy reducers pass exactly one tile tensor. Multi-input reducers
            may pass multiple tensors, and should document expected arity and
            per-input shapes. This base implementation preserves only the legacy
            one-input passthrough behavior.
        mask : torch.Tensor | None, default=None
            Unused placeholder for API parity with non-streaming reducers.
            Behaves as keyword-only metadata and is not part of ``*inputs``.

        Returns
        -------
        torch.Tensor
            Input tensor unchanged for one-input legacy reducers.
        """
        if len(inputs) != 1:
            raise ValueError(
                "BaseStreamingGlobalReducer legacy passthrough expects exactly one tensor input; "
                f"got {len(inputs)}."
            )
        x = inputs[0]
        self._last_output = x
        return x

class StreamingReducer(BaseStreamingGlobalReducer, ABC):
    """Backward-compatible abstract alias for custom streaming reducers.

    This class intentionally does not implement a concrete reduction strategy.
    Subclassers should implement the abstract methods from
    :class:`BaseStreamingGlobalReducer` directly.
    """
