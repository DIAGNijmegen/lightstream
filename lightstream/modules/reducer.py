import torch
import torch.nn as nn


class StreamingReducerTileF(torch.autograd.Function):
    """Tile-local reducer op used by :class:`StreamingReducer`.

    The op supports an optional 2D valid mask and optional normalization factor.
    This lets SCNN keep tile orchestration while reducer math (forward/backward)
    lives in reducer logic.
    """

    @staticmethod
    def forward(
        ctx,
        tile_output: torch.Tensor,
        valid_mask: torch.Tensor | None,
        normalization: torch.Tensor | None,
    ) -> torch.Tensor:
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
            norm = normalization.to(device=tile_output.device, dtype=torch.float32).clamp_min(1)
            reduced_fp32 = masked.sum(dim=(-2, -1), keepdim=True, dtype=torch.float32) / norm
            reduced = reduced_fp32.to(dtype=tile_output.dtype)
            ctx.normalization = norm
            ctx.has_normalization = True
        else:
            reduced = masked.sum(dim=(-2, -1), keepdim=True)
            ctx.normalization = None
            ctx.has_normalization = False

        return reduced

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mask_4d,) = ctx.saved_tensors

        grad_input = grad_output
        if ctx.has_normalization:
            grad_input = (grad_input.to(dtype=torch.float32) / ctx.normalization).to(dtype=grad_output.dtype)

        grad_input = grad_input.expand(-1, -1, ctx.input_height, ctx.input_width)

        if ctx.has_mask:
            grad_input = grad_input * mask_4d.to(dtype=grad_input.dtype, device=grad_input.device)

        return grad_input, None, None


streaming_reduce_tile = StreamingReducerTileF.apply


class Reducer(nn.Module):
    """Global spatial reducer for NCHW tensors."""

    def __init__(self, mode: str = "mean"):
        super().__init__()
        if mode not in {"sum", "mean"}:
            raise ValueError(f"Unsupported reducer mode '{mode}', expected 'sum' or 'mean'.")
        self.mode = mode
        self._streaming_passthrough = False

    def _normalize_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 2:
            if mask.shape != x.shape[-2:]:
                raise ValueError(f"2D mask shape must match (H,W)={tuple(x.shape[-2:])}, got {tuple(mask.shape)}")
            return mask[None, None]

        if mask.ndim == 4:
            if mask.shape[-2:] != x.shape[-2:]:
                raise ValueError(f"4D mask shape must match spatial dims (H,W)={tuple(x.shape[-2:])}, got {tuple(mask.shape)}")
            if mask.shape[0] not in {1, x.shape[0]}:
                raise ValueError(f"4D mask batch dimension must be 1 or N={x.shape[0]}, got {mask.shape[0]}")
            if mask.shape[1] not in {1, x.shape[1]}:
                raise ValueError(f"4D mask channel dimension must be 1 or C={x.shape[1]}, got {mask.shape[1]}")
            return mask

        raise ValueError(f"mask must be 2D (H,W) or 4D (N,C,H,W)/(N,1,H,W), got shape={tuple(mask.shape)}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Reducer expects NCHW tensor, got shape={tuple(x.shape)}")
        if self._streaming_passthrough:
            return x
        if mask is None:
            if self.mode == "sum":
                return x.sum(dim=(-2, -1), keepdim=True)
            return x.mean(dim=(-2, -1), keepdim=True)

        mask_4d = self._normalize_mask(x, mask).to(dtype=x.dtype, device=x.device)
        masked = x * mask_4d

        if self.mode == "sum":
            return masked.sum(dim=(-2, -1), keepdim=True)

        numerator = masked.sum(dim=(-2, -1), keepdim=True, dtype=torch.float32)
        denominator = mask_4d.sum(dim=(-2, -1), keepdim=True, dtype=torch.float32).clamp_min(1)
        return (numerator / denominator).to(dtype=x.dtype)


class StreamingReducer(nn.Module):
    """Streaming counterpart of :class:`Reducer`.

    Responsibility split:
    - SCNN orchestrates tile traversal/placement and decides which tile pixels
      are valid contributors.
    - StreamingReducer owns tile-local reducer math via ``reduce_tile``
      (custom autograd op) and keeps stream accumulation state.
    """

    def __init__(self, mode: str = "mean"):
        super().__init__()
        if mode not in {"sum", "mean"}:
            raise ValueError(f"Unsupported reducer mode '{mode}', expected 'sum' or 'mean'.")
        self.mode = mode
        self._streaming_passthrough = False
        self.register_buffer("running_sum", torch.zeros(0), persistent=False)
        self.register_buffer("running_count", torch.zeros(0), persistent=False)
        self.register_buffer("_stream_seen_mask", torch.zeros(0, dtype=torch.bool), persistent=False)
        self._last_output = None
        self._debug_replay_enabled = False
        self._replay_assignments: list[tuple] | None = None
        self._replay_cursor: int | None = None
        self._forward_tile_masks: list[torch.Tensor] = []
        self._backward_mask_cursor: int = 0

    @classmethod
    def from_reducer(cls, module: Reducer) -> "StreamingReducer":
        return cls(mode=module.mode)

    def to_reducer(self) -> Reducer:
        return Reducer(mode=self.mode)

    def reset_stream_state(
        self,
        batch_size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
        accumulator_dtype: torch.dtype = torch.float32,
    ):
        self.running_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=dtype)
        self.running_count = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=accumulator_dtype)

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
        self.reset_stream_state(batch_size=batch_size, channels=channels, device=device, dtype=dtype)
        self._stream_seen_mask = torch.zeros((output_height, output_width), dtype=torch.bool, device=device)
        self._debug_replay_enabled = debug_replay
        self._replay_assignments = [] if debug_replay else None
        self._replay_cursor = None
        self._forward_tile_masks = []
        self._backward_mask_cursor = 0

    def accumulate_stream_tile(
        self,
        trimmed_output: torch.Tensor,
        tile_y: int,
        tile_x: int,
        sides,
        dst_box,
        user_mask: torch.Tensor | None = None,
    ):
        dst_y0, dst_y1, dst_x0, dst_x1 = dst_box
        seen_slice = self._stream_seen_mask[dst_y0:dst_y1, dst_x0:dst_x1]
        new_mask = ~seen_slice
        valid_mask = new_mask if user_mask is None else (new_mask & user_mask.to(dtype=torch.bool, device=new_mask.device))

        self._forward_tile_masks.append(valid_mask.detach().clone())

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
                    int(trimmed_output.shape[-2]),
                    int(trimmed_output.shape[-1]),
                    int(dst_y0),
                    int(dst_y1),
                    int(dst_x0),
                    int(dst_x1),
                )
            )

        if torch.any(valid_mask):
            self.accumulate_tile(trimmed_output, valid_mask=valid_mask)
            seen_slice |= valid_mask

    def finish_stream(self) -> torch.Tensor:
        return self.finalize_stream()

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
            raise RuntimeError(
                f"Reducer assignment replay incomplete for head {head_idx}: "
                f"consumed={self._replay_cursor}, expected={len(self._replay_assignments)}"
            )

    def accumulate_tile(self, tile_valid_output: torch.Tensor, valid_mask: torch.Tensor | None = None):
        if self.running_sum.numel() == 0:
            self.reset_stream_state(
                batch_size=tile_valid_output.shape[0],
                channels=tile_valid_output.shape[1],
                device=tile_valid_output.device,
                dtype=tile_valid_output.dtype,
            )

        tile_contribution = self.reduce_tile(tile_valid_output, valid_mask=valid_mask)
        self.running_sum = self.running_sum + tile_contribution

        if valid_mask is None:
            n_pixels = tile_valid_output.shape[-1] * tile_valid_output.shape[-2]
        else:
            n_pixels = int(valid_mask.sum().item())

        if self.mode == "mean":
            pixel_increment = torch.tensor(n_pixels, device=self.running_count.device, dtype=self.running_count.dtype)
            self.running_count = self.running_count + pixel_increment

    def finalize_stream(self) -> torch.Tensor:
        if self.running_sum.numel() == 0:
            raise RuntimeError("StreamingReducer state is empty, accumulate_tile() was not called.")
        if self.mode == "sum":
            return self.running_sum
        denom = self.running_count.to(dtype=torch.float32).clamp_min(1)
        if denom.dtype != self.running_sum.dtype:
            denom = denom.to(dtype=self.running_sum.dtype)
        return self.running_sum / denom

    def reduce_tile(
        self,
        tile_output: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        normalization: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return streaming_reduce_tile(tile_output, valid_mask, normalization)

    def build_backward_pair(
        self,
        trimmed_output: torch.Tensor,
        gradient: torch.Tensor,
        *,
        input_y: int,
        input_x: int,
        sides,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build reducer-backed backward pair for a single tile output.

        SCNN provides orchestration metadata (tile location/sides) and optional
        replay assignment entries; reducer applies reducer-specific checks and
        tile-local reduction math.
        """
        expected_h = int(trimmed_output.shape[-2])
        expected_w = int(trimmed_output.shape[-1])

        replay_valid_mask = self._next_backward_mask(trimmed_output)

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
            )

        normalization = self.running_count if self.mode == "mean" else None
        reduced_output = self.reduce_tile(trimmed_output, valid_mask=replay_valid_mask, normalization=normalization)
        return reduced_output, gradient

    def _next_backward_mask(self, trimmed_output: torch.Tensor) -> torch.Tensor | None:
        if self._backward_mask_cursor >= len(self._forward_tile_masks):
            raise RuntimeError("Reducer mask replay cursor out of range.")
        mask = self._forward_tile_masks[self._backward_mask_cursor]
        self._backward_mask_cursor += 1
        if mask.shape != trimmed_output.shape[-2:]:
            raise RuntimeError(
                "Reducer mask replay shape mismatch: "
                f"forward={tuple(mask.shape)} backward={tuple(trimmed_output.shape[-2:])}"
            )
        return mask

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
    ) -> int:
        if cursor >= len(assignments):
            raise RuntimeError("Reducer assignment cursor out of range.")

        (
            f_tile_y,
            f_tile_x,
            f_top,
            f_left,
            f_right,
            f_bottom,
            f_h,
            f_w,
            dst_y0,
            dst_y1,
            dst_x0,
            dst_x1,
        ) = assignments[cursor]

        if (
            int(input_y) != int(f_tile_y)
            or int(input_x) != int(f_tile_x)
            or bool(sides.top) != bool(f_top)
            or bool(sides.left) != bool(f_left)
            or bool(sides.right) != bool(f_right)
            or bool(sides.bottom) != bool(f_bottom)
        ):
            raise RuntimeError(
                "Reducer tile replay mismatch: "
                f"forward tile=({f_tile_y},{f_tile_x},{f_top},{f_left},{f_right},{f_bottom}) "
                f"backward tile=({int(input_y)},{int(input_x)},{bool(sides.top)},{bool(sides.left)},{bool(sides.right)},{bool(sides.bottom)})"
            )

        if expected_h != int(f_h) or expected_w != int(f_w):
            raise RuntimeError(
                "Reducer trimmed shape mismatch: "
                f"forward=({f_h},{f_w}) backward=({expected_h},{expected_w})"
            )

        if (dst_y1 - dst_y0) != expected_h or (dst_x1 - dst_x0) != expected_w:
            raise RuntimeError(
                "Reducer assignment mismatch: "
                f"stored=({dst_y0}:{dst_y1},{dst_x0}:{dst_x1}) current=({expected_h},{expected_w})"
            )

        return cursor + 1

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Marker behavior for streaming path; SCNN performs accumulation.
        self._last_output = x
        return x
