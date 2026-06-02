"""Public streaming engine API."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lightstream.core.constructor import StreamingConstructor

from .config import CompiledPlan, StreamingConfig


class StreamingEngine:
    """Compile and execute a model with tiled streaming semantics.

    Example
    -------
    >>> config = StreamingConfig(tile_shape=(1, 3, 512, 512))
    >>> engine = StreamingEngine(model, config)
    >>> engine.compile()
    >>> y = engine.forward(image, mask=mask)
    >>> engine.backward(image, grad, mask=mask)
    """

    def __init__(self, model: nn.Module, config: StreamingConfig):
        self.model = model
        self.config = config
        self.constructor: StreamingConstructor | None = None
        self.plan: CompiledPlan | None = None

    @property
    def stream_network(self) -> nn.Module | None:
        """Return the compiled streaming network, if compilation has run."""
        return None if self.plan is None else self.plan.stream_network

    def compile(self, input_spec: Any = None, cache: dict | None = None) -> CompiledPlan:
        """Compile the wrapped model into a streaming execution plan.

        Parameters
        ----------
        input_spec : Any, optional
            Reserved for future shape-aware compilation. The current engine uses
            ``config.tile_shape`` as the tile input specification.
        cache : dict, optional
            Tile cache previously produced by :meth:`get_tile_cache` or
            :meth:`state_dict`.
        """
        del input_spec
        tile_shape = self.config.tile_shape
        if len(tile_shape) != 4:
            raise ValueError(f"StreamingConfig.tile_shape must be an NCHW tuple, got {tile_shape!r}")
        if tile_shape[2] != tile_shape[3]:
            raise ValueError("StreamingEngine currently requires square spatial tiles")

        self.constructor = StreamingConstructor(
            self.model,
            tile_size=int(tile_shape[2]),
            verbose=self.config.verbose,
            deterministic=self.config.deterministic,
            saliency=self.config.saliency,
            copy_to_gpu=self.config.copy_to_gpu,
            statistics_on_cpu=self.config.statistics_on_cpu,
            normalize_on_gpu=self.config.normalize_on_gpu,
            mean=self.config.mean,
            std=self.config.std,
            tile_cache=cache,
            add_keep_modules=self.config.add_keep_modules,
            before_streaming_init_callbacks=self.config.before_streaming_init_callbacks,
            after_streaming_init_callbacks=self.config.after_streaming_init_callbacks,
        )
        stream_network = self.constructor.prepare_streaming_model()
        self.plan = stream_network.compiled_plan
        self.plan.stream_network = stream_network
        return self.plan

    def _require_stream_network(self) -> nn.Module:
        stream_network = self.stream_network
        if stream_network is None:
            raise RuntimeError("StreamingEngine.compile() must be called before execution")
        return stream_network

    def forward(self, image: torch.Tensor, *, mask: torch.Tensor | None = None, result_device=None):
        """Run a streaming forward pass."""
        stream_network = self._require_stream_network()
        result_on_cpu = result_device is not None and torch.device(result_device).type == "cpu"
        output = stream_network.forward(image, result_on_cpu=result_on_cpu, mask=mask)
        if result_device is None or result_on_cpu:
            return output
        return self._move_output(output, torch.device(result_device))

    def backward(self, image: torch.Tensor, grad, *, mask: torch.Tensor | None = None) -> None:
        """Run a streaming backward pass."""
        self._require_stream_network().backward(image, grad, mask=mask)

    def get_tile_cache(self) -> dict:
        """Return the compiled tile-cache state."""
        return self._require_stream_network().get_tile_cache()

    def load_tile_cache(self, state: dict) -> None:
        """Load tile-cache state into an already compiled engine."""
        self._require_stream_network().load_tile_cache(state)

    def state_dict(self) -> dict:
        """Alias for :meth:`get_tile_cache` for cache-oriented lifecycle usage."""
        return self.get_tile_cache()

    def load_state_dict(self, state: dict) -> None:
        """Alias for :meth:`load_tile_cache`."""
        self.load_tile_cache(state)

    @classmethod
    def _move_output(cls, output, device: torch.device):
        if isinstance(output, torch.Tensor):
            return output.to(device)
        if isinstance(output, tuple):
            return tuple(cls._move_output(item, device) for item in output)
        if isinstance(output, list):
            return [cls._move_output(item, device) for item in output]
        if isinstance(output, dict):
            return {key: cls._move_output(value, device) for key, value in output.items()}
        return output


__all__ = ["CompiledPlan", "StreamingConfig", "StreamingEngine"]
