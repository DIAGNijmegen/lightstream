"""Immutable configuration produced before streaming execution."""

from dataclasses import dataclass
from contextlib import contextmanager
import copy
from typing import Any

import torch

from .geometry import Lost
from .reducers import StaticReducerBinding

OutputSpec = tuple[str, Any]


@contextmanager
def statistics_hooks(runtime):
    """Install statistics hooks and always remove every installed handle."""
    try:
        runtime._add_hooks_for_statistics()
        yield
    finally:
        runtime._remove_hooks()


@contextmanager
def cudnn_flags(*, deterministic: bool = True, benchmark: bool = False):
    """Temporarily select cuDNN flags without leaking process-global state."""
    previous = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous


@contextmanager
def gradient_mode(enabled: bool):
    """Set autograd mode for a probe and restore the caller's mode."""
    with torch.set_grad_enabled(enabled):
        yield


@contextmanager
def temporary_parameters(runtime):
    """Use constant probe parameters while preserving the complete state dict."""
    state = copy.deepcopy(runtime.stream_module.state_dict())
    try:
        runtime._reset_parameters_to_constant()
        yield
    finally:
        runtime.stream_module.load_state_dict(state)


def _temporary_module_attribute(model, module_type, attribute, value):
    previous = [(module, getattr(module, attribute)) for module in model.modules() if isinstance(module, module_type)]
    try:
        for module, _ in previous:
            setattr(module, attribute, value)
        yield
    finally:
        for module, old_value in previous:
            setattr(module, attribute, old_value)


@contextmanager
def reducer_passthrough(runtime, enabled: bool = True):
    """Temporarily expose reducer inputs during statistics probing."""
    from lightstream.core.reducer import BaseReducer

    yield from _temporary_module_attribute(runtime.stream_module, BaseReducer, "_streaming_passthrough", enabled)


@contextmanager
def normalization_passthrough(runtime, enabled: bool = True):
    """Temporarily bypass channel normalization during statistics probing."""
    from lightstream.core.layers.streaminglayernorm import ChannelLayerNorm

    yield from _temporary_module_attribute(
        runtime.stream_module, ChannelLayerNorm, "_streaming_statistics_passthrough", enabled
    )


@contextmanager
def device_movement(runtime, device=None):
    """Move a model for probing and restore each parameter/buffer's device."""
    original_runtime_device = runtime.device
    locations = []
    for module in runtime.stream_module.modules():
        locations.extend((module, "_parameters", name, tensor.device) for name, tensor in module._parameters.items() if tensor is not None)
        locations.extend((module, "_buffers", name, tensor.device) for name, tensor in module._buffers.items() if tensor is not None)
    try:
        if device is not None:
            runtime.stream_module.to(device)
            runtime.device = torch.device(device)
        yield
    finally:
        # Restore tensors individually: a model can legitimately span devices.
        with torch.no_grad():
            for module, collection_name, name, original_device in locations:
                tensor = getattr(module, collection_name)[name]
                tensor.data = tensor.data.to(original_device)
                if tensor.grad is not None:
                    tensor.grad.data = tensor.grad.data.to(original_device)
        runtime.device = original_runtime_device


@dataclass(frozen=True)
class HeadPlan:
    tile_output_shape: tuple[int, ...]
    stride: tuple[int, ...]
    loss: Lost


@dataclass(frozen=True)
class ModulePlan:
    name: str
    module_type: str
    statistics: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class TilePlan:
    input_shape: tuple[int, ...]
    gradient_loss: Lost
    internal_alignment: tuple[int, int]


@dataclass(frozen=True)
class StreamingPlan:
    """Stable setup results shared by forward and backward executors."""

    tile: TilePlan
    heads: tuple[HeadPlan, ...]
    modules: tuple[ModulePlan, ...]
    reducer_heads: tuple[StaticReducerBinding, ...]
    output_structure: OutputSpec
