import torch
import torch.nn as nn


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

    In streaming mode this module acts as a marker and keeps accumulation state
    managed from SCNN's tile loop.
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

    def accumulate_tile(self, tile_valid_output: torch.Tensor):
        if tile_valid_output.ndim != 4:
            raise ValueError(f"StreamingReducer expects NCHW tile, got shape={tuple(tile_valid_output.shape)}")
        if self.running_sum.numel() == 0:
            self.reset_stream_state(
                batch_size=tile_valid_output.shape[0],
                channels=tile_valid_output.shape[1],
                device=tile_valid_output.device,
                dtype=tile_valid_output.dtype,
            )

        self.running_sum = self.running_sum + tile_valid_output.sum(dim=(-2, -1), keepdim=True)
        n_pixels = tile_valid_output.shape[-1] * tile_valid_output.shape[-2]
        if self.mode == "mean":
            self.running_count = self.running_count + n_pixels

    def finalize_stream(self) -> torch.Tensor:
        if self.running_sum.numel() == 0:
            raise RuntimeError("StreamingReducer state is empty, accumulate_tile() was not called.")
        if self.mode == "sum":
            return self.running_sum
        denom = self.running_count.clamp_min(1)
        return self.running_sum / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Marker behavior for streaming path; SCNN performs accumulation.
        self._last_output = x
        return x
