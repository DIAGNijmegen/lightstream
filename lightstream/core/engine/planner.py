"""Setup/probing boundary for the compatibility facade."""

from .configuration import HeadPlan, ModulePlan, StreamingPlan, TilePlan


class StreamingPlanBuilder:
    """Configure a stream module and freeze all executor-facing metadata."""

    def __init__(self, facade):
        self.facade = facade

    def build(self, *, probe: bool = True) -> StreamingPlan:
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
        reducers = tuple(sorted((int(head), tuple(map(int, inputs))) for head, inputs in f._reducer_input_indices.items()))
        return StreamingPlan(
            tile=TilePlan(tuple(f.tile_shape), f.tile_gradient_lost, tuple(f._compute_internal_alignment())),
            heads=heads,
            modules=tuple(modules),
            reducer_heads=reducers,
            output_structure=f._output_spec,
        )
