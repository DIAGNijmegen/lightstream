"""Public streaming engine API."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lightstream.core.constructor import StreamingConstructor

from .adapters import AdapterRegistry, StreamableLayerAdapter
from .config import CompiledPlan, StreamingConfig
from .orchestration import BackwardExecutor, EngineCollaborators, ForwardExecutor, ReducerRuntime, TilePlanner


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

    def __init__(
        self,
        model: nn.Module,
        config: StreamingConfig,
        *,
        tile_planner: TilePlanner | None = None,
        forward_executor: ForwardExecutor | None = None,
        backward_executor: BackwardExecutor | None = None,
        reducer_runtime: ReducerRuntime | None = None,
        adapter_registry: AdapterRegistry | None = None,
    ):
        self.model = model
        self.config = config
        collaborators = EngineCollaborators.create_default()
        self.tile_planner = tile_planner or collaborators.tile_planner
        self.forward_executor = forward_executor or collaborators.forward_executor
        self.backward_executor = backward_executor or collaborators.backward_executor
        self.reducer_runtime = reducer_runtime or collaborators.reducer_runtime
        self.adapter_registry = adapter_registry or collaborators.adapter_registry
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
        self.constructor = self.tile_planner.build_constructor(self.model, self.config, cache=cache)
        stream_network = self.constructor.prepare_streaming_model()
        self.plan = stream_network.compiled_plan
        self.plan.stream_network = stream_network
        self.reducer_runtime.bind_plan(self.plan)
        return self.plan

    def _require_stream_network(self) -> nn.Module:
        stream_network = self.stream_network
        if stream_network is None:
            raise RuntimeError("StreamingEngine.compile() must be called before execution")
        return stream_network

    def forward(self, image: torch.Tensor, *, mask: torch.Tensor | None = None, result_device=None):
        """Run a streaming forward pass."""
        return self.forward_executor.run(self, image, mask=mask, result_device=result_device)

    def backward(self, image: torch.Tensor, grad, *, mask: torch.Tensor | None = None) -> None:
        """Run a streaming backward pass."""
        self.backward_executor.run(self, image, grad, mask=mask)

    def get_tile_cache(self) -> dict:
        """Return the compiled tile-cache state."""
        stream_network = self._require_stream_network()
        cache = stream_network.get_tile_cache()
        self.plan = stream_network.compiled_plan
        return cache

    def load_tile_cache(self, state: dict) -> None:
        """Load tile-cache state into an already compiled engine."""
        stream_network = self._require_stream_network()
        stream_network.load_tile_cache(state)
        self.plan = stream_network.compiled_plan

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


__all__ = [
    "AdapterRegistry",
    "BackwardExecutor",
    "CompiledPlan",
    "ForwardExecutor",
    "ReducerRuntime",
    "StreamableLayerAdapter",
    "StreamingConfig",
    "StreamingEngine",
    "TilePlanner",
]
