"""Composition collaborators for the public streaming engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from .adapters import AdapterRegistry
from .config import CompiledPlan, StreamingConfig

if TYPE_CHECKING:  # pragma: no cover
    from lightstream.core.constructor import StreamingConstructor

    from .api import StreamingEngine


@dataclass
class TilePlanner:
    """Compile-time tile planning collaborator.

    ``StreamingEngine`` delegates tile-shape validation and constructor creation
    here rather than inheriting planning behavior from a base engine class.
    """

    def validate_config(self, config: StreamingConfig) -> None:
        tile_shape = config.tile_shape
        if len(tile_shape) != 4:
            raise ValueError(f"StreamingConfig.tile_shape must be an NCHW tuple, got {tile_shape!r}")
        if tile_shape[2] != tile_shape[3]:
            raise ValueError("StreamingEngine currently requires square spatial tiles")

    def build_constructor(self, model: nn.Module, config: StreamingConfig, cache: dict | None = None) -> StreamingConstructor:
        from lightstream.core.constructor import StreamingConstructor

        self.validate_config(config)
        return StreamingConstructor(
            model,
            tile_size=int(config.tile_shape[2]),
            verbose=config.verbose,
            deterministic=config.deterministic,
            saliency=config.saliency,
            copy_to_gpu=config.copy_to_gpu,
            statistics_on_cpu=config.statistics_on_cpu,
            normalize_on_gpu=config.normalize_on_gpu,
            mean=config.mean,
            std=config.std,
            tile_cache=cache,
            add_keep_modules=config.add_keep_modules,
            before_streaming_init_callbacks=config.before_streaming_init_callbacks,
            after_streaming_init_callbacks=config.after_streaming_init_callbacks,
        )


@dataclass
class ReducerRuntime:
    """Reducer orchestration collaborator for compiled streaming plans."""

    def bind_plan(self, plan: CompiledPlan) -> None:
        """Hook for reducer-runtime setup after compilation.

        Current reducer execution is implemented by the compiled streaming
        network.  This collaborator keeps reducer orchestration as composition
        state owned by the public engine, ready for custom runtimes without a
        ``BaseStreamingEngine`` hierarchy.
        """
        del plan


@dataclass
class ForwardExecutor:
    """Forward-pass executor collaborator."""

    def run(
        self,
        engine: StreamingEngine,
        image: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        result_device=None,
    ):
        stream_network = engine._require_stream_network()
        result_on_cpu = result_device is not None and torch.device(result_device).type == "cpu"
        output = stream_network.forward(image, result_on_cpu=result_on_cpu, mask=mask)
        if result_device is None or result_on_cpu:
            return output
        return engine._move_output(output, torch.device(result_device))


@dataclass
class BackwardExecutor:
    """Backward-pass executor collaborator."""

    def run(self, engine: StreamingEngine, image: torch.Tensor, grad: Any, *, mask: torch.Tensor | None = None) -> None:
        engine._require_stream_network().backward(image, grad, mask=mask)


@dataclass
class EngineCollaborators:
    """Bundle of collaborators owned by ``StreamingEngine``."""

    tile_planner: TilePlanner
    forward_executor: ForwardExecutor
    backward_executor: BackwardExecutor
    reducer_runtime: ReducerRuntime
    adapter_registry: AdapterRegistry

    @classmethod
    def create_default(cls) -> EngineCollaborators:
        return cls(
            tile_planner=TilePlanner(),
            forward_executor=ForwardExecutor(),
            backward_executor=BackwardExecutor(),
            reducer_runtime=ReducerRuntime(),
            adapter_registry=AdapterRegistry(),
        )
