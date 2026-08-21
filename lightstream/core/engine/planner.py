"""Setup/probing boundary for the compatibility facade."""

from .configuration import HeadPlan, ModulePlan, StreamingPlan, TilePlan
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
            self.facade._configure_legacy()
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
