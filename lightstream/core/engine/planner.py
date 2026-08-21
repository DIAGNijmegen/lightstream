"""Setup/probing boundary for the compatibility facade."""

from contextlib import ExitStack

import torch

from .configuration import (
    HeadPlan, ModulePlan, StreamingPlan, TilePlan, cudnn_flags,
    device_movement, gradient_mode, normalization_passthrough,
    reducer_passthrough, statistics_hooks, temporary_parameters,
)
from .reducers import StaticReducerBinding
from .operators import STREAMING_OPERATORS


class UnsupportedStreamingOperatorError(ValueError):
    """Raised when a leaf module does not declare the streaming contract."""


class StreamingPlanBuilder:
    """Configure a stream module and freeze all executor-facing metadata."""

    def __init__(self, facade):
        self.facade = facade

    def build(self, *, probe: bool = True) -> StreamingPlan:
        self._validate_operators()
        if probe:
            self._probe()
        f = self.facade
        modules = []
        for name, module in f.stream_module.named_modules():
            if module in f._module_stats:
                stats = tuple(sorted(f._module_stats[module].items(), key=lambda item: item[0]))
                modules.append(ModulePlan(name, type(module).__qualname__, stats))
        heads = tuple(
            HeadPlan(tuple(shape), tuple(int(x) for x in stride), loss)
            for shape, stride, loss in zip(f._tile_output_shapes, f._output_stride_per_output, f._tile_output_lost)
        )
        reducers = tuple(
            StaticReducerBinding(name=name, reducer_type=type(module).__qualname__)
            for name, module in f.stream_module.named_modules()
            if module in f._streaming_reducers
        )
        return StreamingPlan(
            tile=TilePlan(tuple(f.tile_shape), f.tile_gradient_lost, tuple(f._compute_internal_alignment())),
            heads=heads,
            modules=tuple(modules),
            reducer_heads=reducers,
            output_structure=f._output_spec,
        )

    def _probe(self) -> None:
        """Probe geometry while confining every temporary configuration change."""
        f = self.facade
        probe_device = torch.device("cpu") if f.statistics_on_cpu else None
        f._stats_per_grad_fn = {}
        try:
            with ExitStack() as stack:
                stack.enter_context(cudnn_flags(deterministic=True, benchmark=False))
                stack.enter_context(temporary_parameters(f))
                stack.enter_context(device_movement(f, probe_device))
                stack.enter_context(statistics_hooks(f))
                stack.enter_context(reducer_passthrough(f))
                stack.enter_context(normalization_passthrough(f))

                tile = torch.ones(f.tile_shape, dtype=f.dtype, requires_grad=True, device=f.device)
                with gradient_mode(False):
                    f._gather_forward_statistics(tile)
                f._print_verbose("")
                with gradient_mode(True):
                    f._gather_backward_statistics(tile)

            # Public structure must be sampled with the restored public behavior.
            with gradient_mode(False), device_movement(f, probe_device):
                f._capture_public_output_spec()
        finally:
            f._remove_hooks()
            f._saved_tensors = {}
            f.__dict__.pop("_stats_per_grad_fn", None)
            for parameter in f.stream_module.parameters():
                if parameter.grad is not None:
                    parameter.grad.detach_()
                    parameter.grad.zero_()

        f._streaming_reducers = []
        f.stream_module = f._convert_modules_for_streaming(f.stream_module)
        f._add_hooks_for_streaming()

    def _validate_operators(self) -> None:
        # Reducers have their own lifecycle protocol and containers merely group
        # operators, so only ordinary leaves require an operator registration.
        from lightstream.core.reducer import BaseReducer, BaseStreamingGlobalReducer

        def leaves(module, prefix=""):
            adapter = STREAMING_OPERATORS.adapter_for(module)
            if adapter is not None or isinstance(module, (BaseReducer, BaseStreamingGlobalReducer)):
                yield prefix, module
                return
            children = tuple(module.named_children())
            if not children:
                yield prefix, module
                return
            for name, child in children:
                yield from leaves(child, f"{prefix}.{name}" if prefix else name)

        for path, module in leaves(self.facade.stream_module):
            if isinstance(module, (BaseReducer, BaseStreamingGlobalReducer)):
                continue
            adapter = STREAMING_OPERATORS.adapter_for(module)
            qualified_path = path or "<root>"
            if adapter is None:
                raise UnsupportedStreamingOperatorError(
                    f"Unsupported streaming operator at '{qualified_path}' "
                    f"({type(module).__module__}.{type(module).__qualname__}): "
                    "missing capability 'conversion' (no registry adapter)"
                )
            if not adapter.capabilities.conversion:
                raise UnsupportedStreamingOperatorError(
                    f"Unsupported streaming operator at '{qualified_path}' "
                    f"({type(module).__module__}.{type(module).__qualname__}): "
                    "missing capability 'conversion'"
                )
