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
            norm = normalization.to(device=tile_output.device, dtype=tile_output.dtype).clamp_min(1)
            reduced = masked.sum(dim=(-2, -1), keepdim=True) / norm
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
            grad_input = grad_input / ctx.normalization

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Reducer expects NCHW tensor, got shape={tuple(x.shape)}")
        if self._streaming_passthrough:
            return x
        if self.mode == "sum":
            return x.sum(dim=(-2, -1), keepdim=True)
        return x.mean(dim=(-2, -1), keepdim=True)


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
        self._last_output = None

    @classmethod
    def from_reducer(cls, module: Reducer) -> "StreamingReducer":
        return cls(mode=module.mode)

    def to_reducer(self) -> Reducer:
        return Reducer(mode=self.mode)

    def reset_stream_state(self, batch_size: int, channels: int, device: torch.device, dtype: torch.dtype):
        self.running_sum = torch.zeros((batch_size, channels, 1, 1), device=device, dtype=dtype)
        self.running_count = torch.zeros((batch_size, 1, 1, 1), device=device, dtype=dtype)

    def reduce_full_output(self, full_output: torch.Tensor) -> torch.Tensor:
        if full_output.ndim != 4:
            raise ValueError(f"StreamingReducer expects NCHW tensor, got shape={tuple(full_output.shape)}")
        if self.mode == "sum":
            return full_output.sum(dim=(-2, -1), keepdim=True)
        return full_output.mean(dim=(-2, -1), keepdim=True)

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
            self.running_count = self.running_count + n_pixels

    def finalize_stream(self) -> torch.Tensor:
        if self.running_sum.numel() == 0:
            raise RuntimeError("StreamingReducer state is empty, accumulate_tile() was not called.")
        if self.mode == "sum":
            return self.running_sum
        denom = self.running_count.clamp_min(1)
        return self.running_sum / denom

    def reduce_tile(
        self,
        tile_output: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        normalization: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return streaming_reduce_tile(tile_output, valid_mask, normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Marker behavior for streaming path; SCNN performs accumulation.
        self._last_output = x
        return x
