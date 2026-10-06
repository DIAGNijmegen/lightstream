"""Spatial consistency MSE for full-frame and streamed execution."""

from __future__ import annotations

import torch

from .base import BaseStreamingGlobalReducer, streaming_reduce_tile
from .reducer_base import BaseReducer
from .utils import prepare_spatial_mask, resolve_accumulator_dtype


def _validate_input(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError(
            f"MSEReducer expects one NCHW [N,C,H,W] tensor, got {type(x).__name__} "
            f"with shape {getattr(x, 'shape', None)}."
        )
    if not torch.is_floating_point(x):
        raise TypeError(f"MSEReducer requires floating-point values, got {x.dtype}.")


class MSEReducer(BaseReducer):
    """Return the spatial population variance of one NCHW tensor.

    For each sample/channel and valid set V, define a_i = x_i, or
    a_i = softmax_V(x)_i when use_softmax=True. The output is
    sum_i (a_i - sum_j(a_j)/|V|)^2 / |V|, shaped [N,C,1,1].
    The softmax is spatial, not across channels. Empty masks return zero.
    """

    def __init__(
        self,
        use_softmax: bool = False,
        accumulator_dtype: torch.dtype | None = None,
        mask_resize: bool = False,
        mask_resize_mode: str = "nearest",
    ):
        super().__init__()
        self.use_softmax = bool(use_softmax)
        self.accumulator_dtype = accumulator_dtype
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if len(inputs) != 1:
            raise ValueError(f"MSEReducer expects exactly one input, got {len(inputs)}.")
        x = inputs[0]
        _validate_input(x)
        if self._streaming_passthrough:
            self._last_output = x.view_as(x)
            return self._last_output

        dtype = resolve_accumulator_dtype(self.accumulator_dtype, x.dtype)
        values = x.to(dtype=dtype)
        valid = (
            torch.ones((x.shape[0], 1, *x.shape[-2:]), device=x.device, dtype=torch.bool)
            if mask is None
            else prepare_spatial_mask(
                mask, x, mask_resize=self.mask_resize, mask_resize_mode=self.mask_resize_mode
            )
        )
        count = valid.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        safe_count = count.clamp_min(1)

        if self.use_softmax:
            masked = torch.where(valid, values, torch.finfo(dtype).min)
            maximum = masked.amax(dim=(-2, -1), keepdim=True)
            shifted = torch.where(valid, masked - maximum, torch.zeros_like(masked))
            exponentials = torch.where(valid, shifted.exp(), torch.zeros_like(shifted))
            denominator = exponentials.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
            probabilities = exponentials / denominator.clamp_min(torch.finfo(dtype).tiny)
            mean = 1.0 / safe_count
            centered = torch.where(valid, probabilities - mean, torch.zeros_like(probabilities))
        else:
            mean = torch.where(valid, values, torch.zeros_like(values)).sum(
                dim=(-2, -1), keepdim=True, dtype=dtype
            ) / safe_count
            centered = torch.where(valid, values - mean, torch.zeros_like(values))

        return (centered.square().sum(dim=(-2, -1), keepdim=True, dtype=dtype) / safe_count).to(x.dtype)

    def to_streaming(self) -> StreamingMSEReducer:
        return StreamingMSEReducer(
            use_softmax=self.use_softmax,
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
        )


class StreamingMSEReducer(BaseStreamingGlobalReducer):
    """Stream spatial MSE with global moments and exact replay slopes.

    Raw mode combines tile means and centered squared sums (Welford/Chan).
    Softmax mode tracks a global maximum m, Z=sum exp(x-m), and
    Q=sum exp(2*(x-m)). Its MSE is Q/(N*Z^2)-1/N^2.
    """

    def __init__(
        self,
        use_softmax: bool = False,
        accumulator_dtype: torch.dtype | None = None,
        mask_resize: bool = False,
        mask_resize_mode: str = "nearest",
    ):
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
        self.use_softmax = bool(use_softmax)
        self.mask_resize = bool(mask_resize)
        self.mask_resize_mode = mask_resize_mode
        self.register_buffer("running_mean", torch.zeros(0), persistent=False)
        self.register_buffer("running_max", torch.zeros(0), persistent=False)
        self.register_buffer("running_denominator", torch.zeros(0), persistent=False)
        self._output_dtype = torch.float32

    def forward(self, *inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if len(inputs) != 1:
            raise ValueError(f"StreamingMSEReducer expects exactly one input, got {len(inputs)}.")
        _validate_input(inputs[0])
        # Distinct views let SCNN identify independent heads sharing a producer.
        self._last_output = inputs[0].view_as(inputs[0])
        return self._last_output

    def init_reduction_state(
        self, *, batch_size: int, channels: int, device: torch.device,
        dtype: torch.dtype, accumulator_dtype: torch.dtype,
    ) -> None:
        self._output_dtype = dtype
        shape = (batch_size, channels, 1, 1)
        self.running_sum = torch.zeros(shape, device=device, dtype=accumulator_dtype)
        if self.use_softmax:
            self.running_max = torch.full(
                shape, torch.finfo(accumulator_dtype).min, device=device, dtype=accumulator_dtype
            )
            self.running_denominator = torch.zeros(shape, device=device, dtype=accumulator_dtype)
            self.running_mean = torch.zeros(0, device=device, dtype=accumulator_dtype)
        else:
            self.running_mean = torch.zeros(shape, device=device, dtype=accumulator_dtype)
            self.running_max = torch.zeros(0, device=device, dtype=accumulator_dtype)
            self.running_denominator = torch.zeros(0, device=device, dtype=accumulator_dtype)

    def accumulate_valid_tile(self, tile: torch.Tensor, valid_mask: torch.Tensor) -> None:
        tile = self._parse_single_input_payload(tile)
        _validate_input(tile)
        if self.running_sum.numel() == 0:
            self.reset_stream_state(tile.shape[0], tile.shape[1], tile.device, tile.dtype)
        if not torch.any(valid_mask):
            return

        dtype = resolve_accumulator_dtype(self.accumulator_dtype, tile.dtype)
        values = tile.detach().to(dtype)
        valid = valid_mask[None, None].to(device=tile.device, dtype=torch.bool)
        tile_count = valid.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
        new_count = self.running_count + tile_count

        if self.use_softmax:
            masked = torch.where(valid, values, torch.finfo(dtype).min)
            tile_max = masked.amax(dim=(-2, -1), keepdim=True)
            shifted = torch.where(valid, masked - tile_max, torch.zeros_like(masked))
            exponentials = torch.where(valid, shifted.exp(), torch.zeros_like(shifted))
            tile_z = exponentials.sum(dim=(-2, -1), keepdim=True, dtype=dtype)
            tile_q = exponentials.square().sum(dim=(-2, -1), keepdim=True, dtype=dtype)
            new_max = torch.maximum(self.running_max, tile_max)
            old_scale = torch.exp(self.running_max - new_max)
            tile_scale = torch.exp(tile_max - new_max)
            self.running_denominator = self.running_denominator * old_scale + tile_z * tile_scale
            self.running_sum = self.running_sum * old_scale.square() + tile_q * tile_scale.square()
            self.running_max = new_max
        else:
            selected = torch.where(valid, values, torch.zeros_like(values))
            tile_mean = selected.sum(dim=(-2, -1), keepdim=True, dtype=dtype) / tile_count
            centered = torch.where(valid, values - tile_mean, torch.zeros_like(values))
            tile_m2 = centered.square().sum(dim=(-2, -1), keepdim=True, dtype=dtype)
            delta = tile_mean - self.running_mean
            self.running_sum = (
                self.running_sum + tile_m2
                + delta.square() * self.running_count * tile_count / new_count
            )
            self.running_mean = self.running_mean + delta * tile_count / new_count

        self.running_count = new_count

    def finalize_from_state(self) -> torch.Tensor:
        if self.running_sum.numel() == 0:
            raise RuntimeError("StreamingMSEReducer state is empty; call start_stream() first.")
        count = self.running_count
        safe_count = count.clamp_min(1)
        if self.use_softmax:
            denominator = torch.where(
                count > 0, self.running_denominator, torch.ones_like(self.running_denominator)
            )
            second_moment = self.running_sum / denominator.square()
            result = (second_moment / safe_count - safe_count.reciprocal().square()).clamp_min(0)
        else:
            result = self.running_sum / safe_count
        return torch.where(count > 0, result, torch.zeros_like(result)).to(self._output_dtype)

    def extra_state_for_backward(self) -> dict[str, torch.Tensor]:
        state = {"count": self.running_count.detach().clamp_min(1)}
        if self.use_softmax:
            denominator = torch.where(
                self.running_count > 0,
                self.running_denominator.detach(),
                torch.ones_like(self.running_denominator),
            )
            state.update({
                "maximum": self.running_max.detach(),
                "denominator": denominator,
                "probability_square_sum": (self.running_sum / denominator.square()).detach(),
            })
        else:
            state["mean"] = self.running_mean.detach()
        return state

    def reduce_tile_for_backward(
        self, trimmed_output: torch.Tensor, valid_mask: torch.Tensor | None, global_context,
    ) -> torch.Tensor:
        tile = self._parse_single_input_payload(trimmed_output)
        _validate_input(tile)
        dtype = resolve_accumulator_dtype(self.accumulator_dtype, tile.dtype)
        values = tile.to(dtype)
        valid = (
            torch.ones((1, 1, *tile.shape[-2:]), device=tile.device, dtype=torch.bool)
            if valid_mask is None
            else valid_mask[None, None].to(device=tile.device, dtype=torch.bool)
        )
        count = global_context["count"].to(device=tile.device, dtype=dtype)

        if self.use_softmax:
            maximum = global_context["maximum"].to(device=tile.device, dtype=dtype)
            denominator = global_context["denominator"].to(device=tile.device, dtype=dtype)
            shifted = torch.where(valid, values - maximum, torch.zeros_like(values))
            probabilities = torch.where(valid, shifted.exp() / denominator, torch.zeros_like(values))
            square_sum = global_context["probability_square_sum"].to(device=tile.device, dtype=dtype)
            slope = 2.0 * probabilities * (probabilities - square_sum) / count
        else:
            mean = global_context["mean"].to(device=tile.device, dtype=dtype)
            slope = 2.0 * (values - mean) / count

        slope = torch.where(valid, slope, torch.zeros_like(slope)).detach()
        # The detached slope is the full-frame derivative at each owned pixel.
        return streaming_reduce_tile(slope * values, valid_mask, None).to(tile.dtype)

    def to_reducer(self) -> MSEReducer:
        return MSEReducer(
            use_softmax=self.use_softmax,
            accumulator_dtype=self.accumulator_dtype,
            mask_resize=self.mask_resize,
            mask_resize_mode=self.mask_resize_mode,
        )
