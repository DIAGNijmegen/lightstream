import torch
import torch.nn as nn


def _normalize_spatial_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Normalize user-provided mask to [N,1,H,W] bool format."""
    if mask.ndim == 2:
        if mask.shape != x.shape[-2:]:
            raise ValueError(f"mask shape {tuple(mask.shape)} must match input spatial shape {tuple(x.shape[-2:])}")
        return mask[None, None].to(device=x.device, dtype=torch.bool)

    if mask.ndim == 3:
        if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"3D mask shape {tuple(mask.shape)} must be [N,H,W] with N={x.shape[0]}, H/W={tuple(x.shape[-2:])}"
            )
        return mask[:, None].to(device=x.device, dtype=torch.bool)

    if mask.ndim == 4:
        if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"4D mask shape {tuple(mask.shape)} must be [N,1,H,W] with N={x.shape[0]}, H/W={tuple(x.shape[-2:])}"
            )
        if mask.shape[1] not in (1, x.shape[1]):
            raise ValueError(
                f"4D mask channel dim must be 1 or C={x.shape[1]}, got {mask.shape[1]}"
            )
        mask_bool = mask.to(device=x.device, dtype=torch.bool)
        if mask_bool.shape[1] == x.shape[1]:
            # collapse channel-wise masks into a single spatial mask to match reducer counting semantics
            mask_bool = torch.any(mask_bool, dim=1, keepdim=True)
        return mask_bool

    raise ValueError(f"mask must be 2D/3D/4D spatial mask, got shape={tuple(mask.shape)}")

def _resolve_accumulator_dtype(accumulator_dtype: torch.dtype | None, reference_dtype: torch.dtype) -> torch.dtype:
    """Resolve reduction accumulator dtype with a minimum precision of float32."""
    if accumulator_dtype is None:
        resolved = reference_dtype if reference_dtype in (torch.float32, torch.float64) else torch.float32
    else:
        resolved = accumulator_dtype
    if resolved not in (torch.float32, torch.float64):
        raise ValueError(
            f"Unsupported accumulator_dtype '{resolved}'. Use torch.float32 or torch.float64."
        )
    return resolved


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
            norm = normalization.to(device=tile_output.device)
            acc_dtype = _resolve_accumulator_dtype(norm.dtype, tile_output.dtype)
            norm = norm.to(dtype=acc_dtype).clamp_min(1)
            reduced_acc = masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype) / norm
            reduced = reduced_acc.to(dtype=tile_output.dtype)
            ctx.normalization = norm
            ctx.has_normalization = True
        else:
            acc_dtype = _resolve_accumulator_dtype(None, tile_output.dtype)
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


class Reducer(nn.Module):
    """Global spatial reducer for NCHW tensors."""

    def __init__(self, mode: str = "mean", accumulator_dtype: torch.dtype | None = None):
        super().__init__()
        if mode not in {"sum", "mean"}:
            raise ValueError(f"Unsupported reducer mode '{mode}', expected 'sum' or 'mean'.")
        self.mode = mode
        self.accumulator_dtype = accumulator_dtype
        self._streaming_passthrough = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Reducer expects NCHW tensor, got shape={tuple(x.shape)}")
        if self._streaming_passthrough:
            return x
        if mask is not None:
            mask_nchw = _normalize_spatial_mask(mask, x)
            masked = x * mask_nchw.to(dtype=x.dtype)
            acc_dtype = _resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
            if self.mode == "sum":
                return masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)
            denom = mask_nchw.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).clamp_min(1)
            mean = masked.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype) / denom
            return mean.to(dtype=x.dtype)
        acc_dtype = _resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        if self.mode == "sum":
            return x.sum(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)
        return x.mean(dim=(-2, -1), keepdim=True, dtype=acc_dtype).to(dtype=x.dtype)


class StreamingReducer(nn.Module):
    """Streaming counterpart of :class:`Reducer`.

    Responsibility split:
    - SCNN orchestrates tile traversal/placement and decides which tile pixels
      are valid contributors.
    - StreamingReducer owns tile-local reducer math via ``reduce_tile``
      (custom autograd op) and keeps stream accumulation state.
    """

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

    @classmethod
    def from_reducer(cls, module: Reducer) -> "StreamingReducer":
        return cls(mode=module.mode, accumulator_dtype=module.accumulator_dtype)

    def to_reducer(self) -> Reducer:
        return Reducer(mode=self.mode, accumulator_dtype=self.accumulator_dtype)

    def reset_stream_state(
        self,
        batch_size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
        accumulator_dtype: torch.dtype | None = None,
    ):
        resolved_acc_dtype = _resolve_accumulator_dtype(accumulator_dtype or self.accumulator_dtype, dtype)
        self.running_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=dtype)
        self.running_count = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=resolved_acc_dtype)

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
                    int(trimmed_output.shape[-2]),
                    int(trimmed_output.shape[-1]),
                    int(dst_y0),
                    int(dst_y1),
                    int(dst_x0),
                    int(dst_x1),
                )
            )

        if torch.any(effective_mask):
            self.accumulate_tile(trimmed_output, valid_mask=effective_mask)
        seen_slice |= new_mask

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
        acc_dtype = _resolve_accumulator_dtype(self.accumulator_dtype, self.running_sum.dtype)
        denom = self.running_count.to(dtype=acc_dtype).clamp_min(1)
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
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build reducer-backed backward pair for a single tile output.

        SCNN provides orchestration metadata (tile location/sides) and optional
        replay assignment entries; reducer applies reducer-specific checks and
        tile-local reduction math.
        """
        expected_h = int(trimmed_output.shape[-2])
        expected_w = int(trimmed_output.shape[-1])

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
        reduced_output = self.reduce_tile(trimmed_output, valid_mask=valid_mask, normalization=normalization)
        return reduced_output, gradient

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
        # Accept optional mask kwarg for API parity with Reducer and model codepaths.
        self._last_output = x
        return x
