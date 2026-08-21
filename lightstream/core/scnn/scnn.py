"""
Author: Hans Pinckaers
MIT License
"""
import math
import copy
import logging
from typing import List

import numpy as np
import torch
import torch.autograd
import torch.backends
import torch.nn.functional

from lightstream.core.engine.geometry import (
    Sides,
    Box,
    Lost,
    _ntuple,
    _new_value_indices,
    B_DIM,
    C_DIM,
    H_DIM,
    W_DIM,
    tile_grid,
    iter_tiles,
)
from lightstream.core.layers.streamingconv import StreamingConv2d
from lightstream.core.layers.streamingupsample import StreamingUpsample2d
from lightstream.core.layers.streaminglayernorm import (
    ChannelLayerNorm,
    StreamingChannelLayerNorm,
)
from lightstream.core.layers.streaminglayerscale import LayerScale, StreamingLayerScale
from lightstream.core.layers.statisticsprobe import StatisticsProbe
from lightstream.core.layers.streamingmerge import StreamingMerge
from lightstream.core.reducer import BaseReducer, BaseStreamingGlobalReducer
from lightstream.core.engine.executors import BackwardCall, BackwardExecutor, ForwardCall, ForwardExecutor
from lightstream.core.engine.planner import StreamingPlanBuilder
from lightstream.core.engine.session import StreamSession
from lightstream.core.engine.reducers import ReducerCoordinator


logger = logging.getLogger(__name__)

_triple = _ntuple(3)


BACKWARD_STREAMING_MODULE_TYPES = (
    StreamingConv2d,
    StreamingUpsample2d,
    StreamingChannelLayerNorm,
    StreamingLayerScale,
)


def _is_spatial_preserving_pointwise_module(module):
    """Return True for pointwise channel modules that preserve spatial support."""
    return isinstance(
        module,
        (
            ChannelLayerNorm,
            StreamingChannelLayerNorm,
            LayerScale,
            StreamingLayerScale,
            StatisticsProbe,
            StreamingMerge,
        ),
    )


def _is_backward_streaming_module(module):
    """Return True for streaming modules that need backward tile location state."""
    return isinstance(module, BACKWARD_STREAMING_MODULE_TYPES)


class StreamingCNN(torch.nn.Module):
    """Initialize Streaming CNN helper class. After initialization use the
    forward() and backward() function of this class to lightstream.
    Pseudocode example:

    ```python
    sCNN = StreamingCNN(stream_layers, tile_shape=(1, 3, 600, 600))
    str_output = sCNN.forward(image)
    final_output = final_layers(str_output)
    loss = criterion(final_output, labels)
    loss.backward()
    sCNN.backward(image, str_output.grad)
    ```

    Hooks are used to perform lightstream, to use the stream_layers without
    lightstream you can disable StreamingCNN with the disable() function.
    Subsequently, enable() enables it again. Streaming gets enabled by default
    after initialization.

    Pipeline overview:
      1. Collect per-layer forward/backward tile statistics during configuration.
      2. Convert supported layers to streaming variants and run tiled forward.
      3. Stitch non-reducer heads directly while reducer heads accumulate stream state.
      4. Replay the same tile traversal for backward and build per-head backward pairs.

    Invariants:
      - Flattened output structure from backward gradients must match forward output spec.
      - Reducer replay (when debug enabled) must consume exactly the recorded forward
        assignments in identical order.
      - Every output head must be populated after forward stream completion.
    """

    def __init__(
        self,
        stream_module,
        tile_shape,
        verbose=False,
        deterministic=False,
        saliency=False,
        eps=1e-5,
        copy_to_gpu=True,
        dtype=None,
        statistics_on_cpu=False,
        normalize_on_gpu=False,
        mean=None,
        std=None,
        state_dict=None,
    ):
        """
        Parameters:
            stream_module (torch.nn.Module): modules containing the to be streamed layers
            tile_shape (tuple, NCHW): size of the to be streamed tiles
            verbose (bool): if True, print setup-time tile statistics and progress output
                during configuration. Per-forward tiling diagnostics are emitted through this
                module's logger at DEBUG level (default is False).
            deterministic (bool): whether to use the deterministic algorithms for cudnn
            saliency (bool): will gather the gradients of the input image (saliency map)
            eps (float): epsilon error to compare floating values
        """
        super().__init__()
        global H_DIM, W_DIM
        self.stream_module = stream_module
        self.verbose = verbose
        self.deterministic = deterministic
        self.eps = eps
        self.device = next(stream_module.parameters()).device
        self.dtype = next(stream_module.parameters()).dtype
        if dtype is not None:
            self.dtype = dtype
        self.tile_shape = tile_shape
        self.gather_input_gradient = saliency
        self.copy_to_gpu = copy_to_gpu
        self.statistics_on_cpu = statistics_on_cpu

        if mean is not None and not isinstance(mean, torch.Tensor):
            mean = torch.Tensor(mean)[:, None, None]

        if std is not None and not isinstance(std, torch.Tensor):
            std = torch.Tensor(std)[:, None, None]

        self.mean = mean if mean is not None else torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = std if std is not None else torch.tensor([0.229, 0.224, 0.225])[:, None, None]

        self.should_normalize = normalize_on_gpu

        self._tile_output_shape = None
        self._tile_output_shapes = None
        self._tile_output_lost = None
        self._output_stride_per_output = None
        self._output_spec = None
        self._module_stats = {}
        self._configuration_saved_tensors = {}
        self.debug_reducer_replay = False
        self.debug_forward_sentinel_check = False
        self.debug_backward_tile_alignment = False
        self._hooks = []
        self._streaming_reducers = []

        builder = StreamingPlanBuilder(self)
        if state_dict is None:
            self.plan = builder.build()
        else:
            self.load_tile_cache(state_dict)
            self.plan = builder.build(probe=False)
        self._forward_executor = ForwardExecutor(self)
        self._backward_executor = BackwardExecutor(self, _is_backward_streaming_module)
        self.reducer_coordinator = ReducerCoordinator(self)

    @property
    def _session(self):
        """Return executor-owned invocation state (or ``None`` during setup)."""
        backward = getattr(self, "_backward_executor", None)
        if backward is not None and backward.executing_session is not None:
            return backward.executing_session
        forward = getattr(self, "_forward_executor", None)
        if forward is None:
            return None
        return forward.executing_session or forward.pending_session or forward.last_session

    def _session_field(self, name):
        session = self._session
        if session is None:
            raise RuntimeError(f"Invocation state '{name}' is unavailable outside a streaming session.")
        return getattr(session, name)

    def _set_session_field(self, name, value):
        session = self._session
        if session is None:
            # Compatibility for diagnostic helpers that are exercised without
            # a full forward; normal invocation state is always executor-created.
            session = StreamSession((), self.dtype)
            self._forward_executor.last_session = session
        setattr(session, name, value)

    _active_reducer_mask = property(lambda self: self._session_field("active_reducer_mask"), lambda self, v: self._set_session_field("active_reducer_mask", v))
    _active_reducer_mask_image = property(lambda self: self._session_field("active_reducer_mask_image"), lambda self, v: self._set_session_field("active_reducer_mask_image", v))
    _prepared_reducer_domain_masks = property(lambda self: self._session_field("prepared_reducer_domain_masks"), lambda self, v: self._set_session_field("prepared_reducer_domain_masks", v))
    _current_output_heights = property(lambda self: self._session_field("output_heights"), lambda self, v: self._set_session_field("output_heights", v))
    _current_output_widths = property(lambda self: self._session_field("output_widths"), lambda self, v: self._set_session_field("output_widths", v))
    _last_forward_tiles = property(lambda self: self._session_field("forward_tiles"), lambda self, v: self._set_session_field("forward_tiles", v))
    _reducer_head_map = property(lambda self: self._session_field("reducer_head_map"), lambda self, v: self._set_session_field("reducer_head_map", v))
    _reducer_input_indices = property(lambda self: self._session_field("reducer_input_indices"), lambda self, v: self._set_session_field("reducer_input_indices", v))
    saliency_map = property(lambda self: self._session_field("saliency_map"), lambda self, v: self._set_session_field("saliency_map", v))
    saliency_old_indices = property(lambda self: self._session_field("saliency_old_indices"), lambda self, v: self._set_session_field("saliency_old_indices", v))

    @property
    def _saved_tensors(self):
        session = self._session
        return self._configuration_saved_tensors if session is None else session.saved_tensors

    @_saved_tensors.setter
    def _saved_tensors(self, value):
        session = self._session
        if session is None:
            self._configuration_saved_tensors = value
        else:
            session.saved_tensors = value

    def _print_verbose(self, *args: object, **kwargs: object) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def _configure_legacy(self):
        # Save current model and cudnn flags, since we need to change them and restore later
        state_dict = self._save_parameters()
        (
            old_deterministic_flag,
            old_benchmark_flag,
        ) = self._set_cudnn_flags_to_determistic()
        self._reset_parameters_to_constant()

        # Add hooks to each layer to gather statistics
        self._add_hooks_for_statistics()
        self._set_reducer_passthrough(True)
        self._set_channel_layer_norm_statistics_passthrough(True)

        # We need to temporary store statistics per layer to keep track of the
        # total output stride at each layer
        self._stats_per_grad_fn = {}

        # TODO; temp hack for tile sizes too big on gpu,
        # we need float32 precision
        if self.statistics_on_cpu:
            self.stream_module = self.stream_module.cpu()
            self.device = torch.device("cpu")  # type:ignore

        # Create all-ones tile
        tile = torch.ones(self.tile_shape, dtype=self.dtype, requires_grad=True, device=self.device)

        self._gather_forward_statistics(tile)
        self._print_verbose("")
        self._gather_backward_statistics(tile)

        # TODO; temp hack for tile sizes too big on gpu,
        if self.statistics_on_cpu:
            self.stream_module = self.stream_module.cuda()
            self.device = torch.device("cuda")  # type:ignore

        # Remove all hooks and add hooks for correcting gradients
        # during lightstream
        self._remove_hooks()
        self._set_reducer_passthrough(False)
        self._set_channel_layer_norm_statistics_passthrough(False)
        #
        self._restore_parameters(state_dict)
        self._capture_public_output_spec()
        self._streaming_reducers = []
        self.stream_module = self._convert_modules_for_streaming(self.stream_module)
        self._add_hooks_for_streaming()

        # Remove temporary data
        self._saved_tensors = {}
        del self._stats_per_grad_fn

        # Zero the gradients
        for param in self.stream_module.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

        self._set_cudnn_flags(old_deterministic_flag, old_benchmark_flag)
        del state_dict

    def _capture_public_output_spec(self) -> None:
        """Capture user-facing output structure with reducers in normal (non-passthrough) mode."""
        spec_tile = torch.ones(self.tile_shape, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output = self.stream_module(spec_tile)
        _, output_spec = self._flatten_output_structure(output)
        self._output_spec = output_spec

    def _set_reducer_passthrough(self, enabled: bool):
        for mod in self.stream_module.modules():
            if isinstance(mod, BaseReducer):
                mod._streaming_passthrough = enabled

    def _set_channel_layer_norm_statistics_passthrough(self, enabled: bool):
        for mod in self.stream_module.modules():
            if isinstance(mod, ChannelLayerNorm):
                mod._streaming_statistics_passthrough = enabled

    def _gather_backward_statistics(self, tile):
        # Forward pass with grads enabled
        torch.set_grad_enabled(True)
        output = self.stream_module(tile)
        output_tensors, output_spec = self._flatten_output_structure(output)
        self._output_spec = output_spec

        # Gather backward statistics
        self._tile_output_shapes = [out.shape for out in output_tensors]
        self._tile_output_shape = self._tile_output_shapes[0]
        self._output_stride_per_output = []
        gradients = []
        for idx, out in enumerate(output_tensors):
            lost = self._tile_output_lost[idx]
            gradient = torch.zeros(*out.shape, dtype=self.dtype, device=self.device)
            gradient[
                :,
                :,
                lost.top : out.shape[H_DIM] - lost.bottom,
                lost.left : out.shape[W_DIM] - lost.right,
            ] = 1
            gradients.append(gradient)

            p_stats = self._prev_stats(out)
            if p_stats:
                output_stride = p_stats["output_stride"] * torch.tensor(p_stats["stride"])
            else:
                output_stride = torch.tensor([1, 1, 1])

            self._output_stride_per_output.append(output_stride)

        self.output_stride = self._output_stride_per_output[0]
        self._base_output_stride = self._output_stride_per_output[0].clone()
        for stride in self._output_stride_per_output[1:]:
            self._base_output_stride[1] = min(int(self._base_output_stride[1]), int(stride[1]))
            self._base_output_stride[2] = min(int(self._base_output_stride[2]), int(stride[2]))
        torch.autograd.backward(output_tensors, gradients)

        # tiles can have -1, see backward_statistics_hook
        self.tile_gradient_lost = self._non_max_border_amount(tile.grad)

        # lost statistics assume you're always in the middle of an image, so left,bottom,top,right lost can always happen
        self._print_verbose("\n", "Input gradient lost", self.tile_gradient_lost)

    def _gather_forward_statistics(self, tile):
        torch.set_grad_enabled(False)
        output = self.stream_module(tile)
        output_tensors, output_spec = self._flatten_output_structure(output)
        self._output_spec = output_spec
        self._tile_output_lost = [self._non_max_border_amount(out) for out in output_tensors]
        self.tile_output_lost = self._tile_output_lost[0]
        self._print_verbose("\n", "Output lost", self._tile_output_lost)

    def _flatten_output_structure(self, output):
        if isinstance(output, torch.Tensor):
            return [output], ("tensor", None)
        if isinstance(output, tuple):
            flat = []
            children = []
            for x in output:
                child_flat, child_spec = self._flatten_output_structure(x)
                flat.extend(child_flat)
                children.append(child_spec)
            return flat, ("tuple", children)
        if isinstance(output, list):
            flat = []
            children = []
            for x in output:
                child_flat, child_spec = self._flatten_output_structure(x)
                flat.extend(child_flat)
                children.append(child_spec)
            return flat, ("list", children)
        if isinstance(output, dict):
            flat = []
            children = []
            for key in sorted(output.keys()):
                child_flat, child_spec = self._flatten_output_structure(output[key])
                flat.extend(child_flat)
                children.append((key, child_spec))
            return flat, ("dict", children)
        raise TypeError(f"Unsupported output type for streaming: {type(output)}")

    def _unflatten_output_structure(self, flat, spec, index=0):
        kind, payload = spec
        if kind == "tensor":
            return flat[index], index + 1
        if kind in {"tuple", "list"}:
            values = []
            for child in payload:
                value, index = self._unflatten_output_structure(flat, child, index)
                values.append(value)
            return (tuple(values) if kind == "tuple" else values), index
        if kind == "dict":
            values = {}
            for key, child in payload:
                value, index = self._unflatten_output_structure(flat, child, index)
                values[key] = value
            return values, index
        raise TypeError(f"Unsupported output spec kind: {kind}")

    def _reducer_aux_indices(self) -> set[int]:
        aux_indices = set()
        for reducer_head, indices in self._reducer_input_indices.items():
            if reducer_head in self._reducer_head_map:
                aux_indices.update(indices[1:])
        return aux_indices

    def _public_output_indices(self) -> list[int]:
        reducer_aux_indices = self._reducer_aux_indices()
        return [idx for idx in range(len(self._tile_output_shapes)) if idx not in reducer_aux_indices]

    def _public_output_debug_context(self, public_indices, reducer_aux_indices=None) -> str:
        if reducer_aux_indices is None:
            reducer_aux_indices = self._reducer_aux_indices()
        return (
            f"public_indices={list(public_indices)}, "
            f"reducer_auxiliary_indices={sorted(reducer_aux_indices)}, "
            f"self._reducer_input_indices={self._reducer_input_indices}"
        )

    def _validate_public_output_indices(self, public_indices) -> None:
        reducer_aux_indices = self._reducer_aux_indices()
        leaked_aux_indices = sorted(set(public_indices) & reducer_aux_indices)
        if leaked_aux_indices:
            raise RuntimeError(
                "Public output indices include reducer auxiliary indices; "
                f"leaked_auxiliary_indices={leaked_aux_indices}; "
                f"{self._public_output_debug_context(public_indices, reducer_aux_indices)}"
            )

    def _validate_public_forward_outputs(self, outputs, public_indices) -> None:
        context = self._public_output_debug_context(public_indices)
        for idx in public_indices:
            output = outputs[idx]
            if output is None:
                raise RuntimeError(f"Public output head {idx} was not populated during streaming forward; {context}")
            if getattr(self, "debug_forward_sentinel_check", False) and torch.all(output == 999):
                raise RuntimeError(
                    f"Public output head {idx} still contains only the unstitched sentinel value 999; {context}"
                )

    def _count_tensors_in_spec(self, spec) -> int:
        kind, payload = spec
        if kind == "tensor":
            return 1
        if kind in {"tuple", "list"}:
            return sum(self._count_tensors_in_spec(child) for child in payload)
        if kind == "dict":
            return sum(self._count_tensors_in_spec(child) for _, child in payload)
        raise TypeError(f"Unsupported output spec kind: {kind}")

    def _compute_internal_safe_input_step(self):
        """Compute conservative input-step bounds from per-layer lost-region stats.

        Upsampling layers report two different losses in layer-local coordinates:
        forward loss is measured on the high-resolution upsample output, while
        backward input loss is measured on the low-resolution gradient input.
        Convert both to input-image coordinates before comparing them to the
        input tile size so tile overlap covers the largest lost border.
        """
        candidates_h = []
        candidates_w = []

        for mod in self.stream_module.modules():
            if isinstance(mod, StreamingConv2d):
                if not hasattr(mod, "grad_lost") or mod.grad_lost is None:
                    continue
                stride = torch.tensor(_triple(mod.stride))
                output_stride = mod.output_stride * stride
                step_h = self.tile_shape[H_DIM] - int(mod.grad_lost.top + mod.grad_lost.bottom) * int(output_stride[1])
                step_w = self.tile_shape[W_DIM] - int(mod.grad_lost.left + mod.grad_lost.right) * int(output_stride[2])
                if step_h > 0 and step_w > 0:
                    candidates_h.append(step_h)
                    candidates_w.append(step_w)
            elif isinstance(mod, StreamingUpsample2d):
                stats = self._module_stats.get(mod, {})
                forward_lost = stats.get("lost")
                backward_input_lost = stats.get(
                    "upsample_backward_input_lost",
                    getattr(mod, "upsample_backward_input_lost", None),
                )
                pre_upsample_output_stride = torch.as_tensor(
                    getattr(mod, "pre_upsample_output_stride", torch.tensor([1, 1, 1])),
                    dtype=torch.long,
                )
                post_upsample_output_stride = torch.as_tensor(
                    getattr(
                        mod,
                        "post_upsample_output_stride",
                        getattr(mod, "output_stride", torch.tensor([1, 1, 1])),
                    ),
                    dtype=torch.long,
                )

                upsample_candidates_h = []
                upsample_candidates_w = []
                if forward_lost is not None:
                    step_h = self.tile_shape[H_DIM] - int(forward_lost.top + forward_lost.bottom) * int(
                        post_upsample_output_stride[1]
                    )
                    step_w = self.tile_shape[W_DIM] - int(forward_lost.left + forward_lost.right) * int(
                        post_upsample_output_stride[2]
                    )
                    if step_h > 0 and step_w > 0:
                        candidates_h.append(step_h)
                        candidates_w.append(step_w)
                        upsample_candidates_h.append(step_h)
                        upsample_candidates_w.append(step_w)
                if backward_input_lost is not None:
                    step_h = self.tile_shape[H_DIM] - int(backward_input_lost.top + backward_input_lost.bottom) * int(
                        pre_upsample_output_stride[1]
                    )
                    step_w = self.tile_shape[W_DIM] - int(backward_input_lost.left + backward_input_lost.right) * int(
                        pre_upsample_output_stride[2]
                    )
                    if step_h > 0 and step_w > 0:
                        candidates_h.append(step_h)
                        candidates_w.append(step_w)
                        upsample_candidates_h.append(step_h)
                        upsample_candidates_w.append(step_w)
                if upsample_candidates_h and upsample_candidates_w:
                    logger.debug(
                        "Upsample safe tile step: %s",
                        {
                            "module": mod,
                            "forward_lost": forward_lost,
                            "backward_input_lost": backward_input_lost,
                            "pre_upsample_output_stride": pre_upsample_output_stride,
                            "post_upsample_output_stride": post_upsample_output_stride,
                            "safe_tile_step": (
                                min(upsample_candidates_h),
                                min(upsample_candidates_w),
                            ),
                        },
                    )

        # Fallback to global backward-safe span if per-layer stats are unavailable
        grad_safe_h = self.tile_shape[H_DIM] - self.tile_gradient_lost.top - self.tile_gradient_lost.bottom
        grad_safe_w = self.tile_shape[W_DIM] - self.tile_gradient_lost.left - self.tile_gradient_lost.right
        candidates_h.append(int(grad_safe_h))
        candidates_w.append(int(grad_safe_w))

        return max(1, min(candidates_h)), max(1, min(candidates_w))

    def _module_alignment_stats(self, module):
        """Return module input-space stride stats used for tile-start alignment."""
        stats = self._module_stats.get(module, {})

        stride = stats.get("stride")
        if stride is None:
            stride = getattr(module, "stride", 1)
            if stride is None:
                stride = getattr(module, "kernel_size", 1)
        stride = torch.as_tensor(_triple(stride), dtype=torch.long)

        output_stride = stats.get("output_stride")
        if output_stride is None:
            output_stride = getattr(module, "output_stride", torch.tensor([1, 1, 1]))
        output_stride = torch.as_tensor(_triple(output_stride), dtype=torch.long)

        return output_stride, stride

    def _compute_internal_alignment(self):
        """Compute input-space alignment constraints from internal downsampling layers.

        When output heads are upsampled back to stride-1, alignment based only on
        head output stride becomes 1 and can lose the internal phase constraints
        required by earlier strided convolutions and pooling layers.
        """

        align_h = 1
        align_w = 1
        alignment_modules = (
            StreamingConv2d,
            torch.nn.Conv2d,
            torch.nn.MaxPool2d,
            torch.nn.AvgPool2d,
        )
        for module in self.stream_module.modules():
            if not isinstance(module, alignment_modules):
                continue

            output_stride, stride = self._module_alignment_stats(module)
            if int(stride[1]) <= 1 and int(stride[2]) <= 1:
                continue

            effective_h = int(output_stride[1]) * int(stride[1])
            effective_w = int(output_stride[2]) * int(stride[2])
            align_h = math.lcm(align_h, max(1, effective_h))
            align_w = math.lcm(align_w, max(1, effective_w))

        return align_h, align_w

    def _compute_multi_output_input_step(self, valid_output_heights, valid_output_widths, include_grad_safe=True):
        step_candidates_h = [
            valid_output_heights[idx] * int(self._output_stride_per_output[idx][1])
            for idx in range(len(self._tile_output_shapes))
        ]
        step_candidates_w = [
            valid_output_widths[idx] * int(self._output_stride_per_output[idx][2])
            for idx in range(len(self._tile_output_shapes))
        ]

        # Extra safety from backward statistics (input gradient valid region)
        if include_grad_safe:
            grad_safe_h, grad_safe_w = self._compute_internal_safe_input_step()
            step_candidates_h.append(int(grad_safe_h))
            step_candidates_w.append(int(grad_safe_w))

        align_h = 1
        align_w = 1
        for stride in self._output_stride_per_output:
            align_h = math.lcm(align_h, int(stride[1]))
            align_w = math.lcm(align_w, int(stride[2]))

        internal_align_h, internal_align_w = self._compute_internal_alignment()
        align_h = math.lcm(align_h, internal_align_h)
        align_w = math.lcm(align_w, internal_align_w)

        valid_input_height = max(align_h, (min(step_candidates_h) // align_h) * align_h)
        valid_input_width = max(align_w, (min(step_candidates_w) // align_w) * align_w)
        return valid_input_height, valid_input_width

    def _convert_modules_for_streaming(self, module):
        mod = module
        if isinstance(module, torch.nn.Conv2d):
            if module in self._module_stats:
                mod = StreamingConv2d.from_torch_conv2d(module)
                mod.grad_lost = self._module_stats[module]["grad_lost"]
                mod.output_stride = self._module_stats[module]["output_stride"]
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, torch.nn.Upsample):
            mod = StreamingUpsample2d.from_torch_upsample(module)
            if module in self._module_stats:
                stats = self._module_stats[module]
                mod.grad_lost = stats.get("grad_lost", Lost(0, 0, 0, 0))
                mod.output_stride = stats.get(
                    "post_upsample_output_stride",
                    stats.get("output_stride", torch.tensor([1, 1, 1])),
                )
                mod.pre_upsample_output_stride = stats.get("pre_upsample_output_stride")
                if mod.pre_upsample_output_stride is None:
                    scale_h, scale_w = stats.get("scale_factor_hw", (mod.scale_factor, mod.scale_factor))
                    if isinstance(scale_h, tuple):
                        scale_h, scale_w = scale_h[-2], scale_h[-1]
                    if scale_h is None or scale_w is None:
                        scale_h, scale_w = 1.0, 1.0
                    mod.pre_upsample_output_stride = mod.output_stride.clone().detach().to(torch.float32)
                    mod.pre_upsample_output_stride[1] *= float(scale_h)
                    mod.pre_upsample_output_stride[2] *= float(scale_w)
                    mod.pre_upsample_output_stride = torch.round(mod.pre_upsample_output_stride).to(torch.long)
                    mod.pre_upsample_output_stride[0] = 1
                    stats["pre_upsample_output_stride"] = mod.pre_upsample_output_stride
                stats.setdefault("post_upsample_output_stride", mod.output_stride)
                for key in (
                    "scale_factor_hw",
                    "pre_upsample_output_stride",
                    "post_upsample_output_stride",
                    "grad_lost",
                    "side_aware_grad_lost",
                    "backward_valid_lost",
                    "upsample_forward_output_lost",
                ):
                    if key in stats and hasattr(mod, key):
                        setattr(mod, key, stats[key])
                if mod.mode == "bilinear" and "upsample_backward_input_lost" in stats:
                    mod.upsample_backward_input_lost = stats["upsample_backward_input_lost"]
                else:
                    mod.upsample_backward_input_lost = None
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, LayerScale):
            mod = StreamingLayerScale.from_layer_scale(module)
            if module in self._module_stats:
                stats = self._module_stats[module]
                if "grad_lost" in stats:
                    mod.grad_lost = stats["grad_lost"]
                if "output_stride" in stats:
                    mod.output_stride = stats["output_stride"]
                self._module_stats[mod] = stats
                del self._module_stats[module]
            del module
            return mod
        elif isinstance(module, ChannelLayerNorm):
            mod = StreamingChannelLayerNorm.from_channel_layer_norm(module)
            if module in self._module_stats:
                stats = self._module_stats[module]
                if "grad_lost" in stats:
                    mod.grad_lost = stats["grad_lost"]
                if "output_stride" in stats:
                    mod.output_stride = stats["output_stride"]
                self._module_stats[mod] = stats
                del self._module_stats[module]
            # ChannelLayerNorm and StreamingChannelLayerNorm both wrap a
            # torch.nn.LayerNorm child internally. Treat the wrapper as a leaf
            # while parent modules continue recursing so conversion preserves
            # compatible state-dict keys like ``norm.weight`` and ``norm.bias``.
            del module
            return mod
        elif isinstance(module, BaseReducer):
            mod = module.to_streaming()
            self._streaming_reducers.append(mod)
        for name, child in module.named_children():
            mod.add_module(name, self._convert_modules_for_streaming(child))
        del module
        return mod

    def _reset_converted_modules(self, module):
        mod = module
        if isinstance(module, StreamingConv2d):
            mod = module.to_torch_conv2d()
            if module not in self._module_stats:
                stats = {}
                stats["grad_lost"] = module.grad_lost
                stats["output_stride"] = module.output_stride
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, StreamingUpsample2d):
            mod = module.to_torch_upsample()
            if module not in self._module_stats:
                stats = {}
                stats["grad_lost"] = module.grad_lost
                stats["pre_upsample_output_stride"] = module.pre_upsample_output_stride
                stats["output_stride"] = module.output_stride
                stats["post_upsample_output_stride"] = module.output_stride
                stats["backward_valid_lost"] = module.backward_valid_lost
                stats["upsample_backward_input_lost"] = module.upsample_backward_input_lost
                stats["upsample_forward_output_lost"] = module.upsample_forward_output_lost
                if module.scale_factor is not None:
                    sf = module.scale_factor
                    stats["scale_factor_hw"] = (
                        (float(sf[-2]), float(sf[-1])) if isinstance(sf, tuple) else (float(sf), float(sf))
                    )
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, StreamingLayerScale):
            mod = module.to_layer_scale()
            if module not in self._module_stats:
                stats = {}
                stats["grad_lost"] = module.grad_lost
                stats["output_stride"] = module.output_stride
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, StreamingChannelLayerNorm):
            mod = module.to_channel_layer_norm()
            if module not in self._module_stats:
                stats = {}
                stats["grad_lost"] = module.grad_lost
                stats["output_stride"] = module.output_stride
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, BaseStreamingGlobalReducer):
            mod = module.to_reducer()
        for name, child in module.named_children():
            mod.add_module(name, self._reset_converted_modules(child))
        del module
        return mod

    def _reset_parameters_to_constant(self):
        for mod in self.stream_module.modules():
            if isinstance(mod, (torch.nn.Conv2d)):
                # to counter floating precision errors, we assign 1 to the weights and
                # normalize the output after the conv.
                kernel_h, kernel_w = mod.kernel_size
                fan_in = (mod.in_channels / mod.groups) * kernel_h * kernel_w
                fan_out = (mod.out_channels / mod.groups) * kernel_h * kernel_w
                scale = 1.0 / max(fan_in, fan_out)
                # Keep positive support geometry while preventing fan-in/fan-out
                # amplification during tile-statistics generation.
                torch.nn.init.constant_(mod.weight, scale)
                if mod.bias is not None:
                    torch.nn.init.constant_(mod.bias, 0)

        # Only BatchNorm has running statistics that can be frozen into a
        # deterministic affine-like transform while gathering tile statistics.
        # ChannelLayerNorm/LayerNorm computes per-sample channel statistics from
        # the current tensor, and its default affine parameters are already
        # weight=1 and bias=0; changing those parameters does not prevent
        # constant setup tensors from producing zero LayerNorm input gradients.
        for m in self.stream_module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.eval()

    def _set_cudnn_flags(self, deterministic_flag, benchmark_flag):
        torch.backends.cudnn.deterministic = deterministic_flag
        torch.backends.cudnn.benchmark = benchmark_flag

    def _set_cudnn_flags_to_determistic(self):
        deterministic_flag = torch.backends.cudnn.deterministic
        benchmark_flag = torch.backends.cudnn.benchmark
        self._set_cudnn_flags(True, False)
        return deterministic_flag, benchmark_flag

    def _save_parameters(self):
        state_dict = self.stream_module.state_dict()
        state_dict = copy.deepcopy(state_dict)
        return state_dict

    def _restore_parameters(self, state_dict):
        self.stream_module.load_state_dict(state_dict)

    def _non_max_border_amount(self, tensor):
        # Sum over the channels, useful for networks that treat certain channels
        # different (e.g., DenseNet)
        if tensor.dim() > 3:
            tensor = torch.sum(tensor, dim=1)[0]
        tensor = tensor / tensor.max()  # normalize
        tensor = tensor > tensor.max() * (1 - self.eps)
        non_zero = tensor.nonzero(as_tuple=False)
        top, left = non_zero.min(dim=0)[0]
        # for bottom and right we need to substract -1: correct index 3 is actually the 4th pixel
        bottom, right = (
            torch.tensor([*tensor.size()], dtype=torch.long, device=self.device) - non_zero.max(dim=0)[0] - 1
        )
        return Lost(int(top), int(left), int(bottom), int(right))

    def _compute_valid_output_sizes(self):
        """Return per-head valid output sizes after removing border loss."""
        valid_output_heights = [
            self._tile_output_shapes[idx][H_DIM] - self._tile_output_lost[idx].top - self._tile_output_lost[idx].bottom
            for idx in range(len(self._tile_output_shapes))
        ]
        valid_output_widths = [
            self._tile_output_shapes[idx][W_DIM] - self._tile_output_lost[idx].left - self._tile_output_lost[idx].right
            for idx in range(len(self._tile_output_shapes))
        ]
        return valid_output_heights, valid_output_widths


    def _compute_valid_input_step(self, valid_output_heights, valid_output_widths):
        """Return input-space stride between neighboring streaming tiles."""
        if len(self._tile_output_shapes) > 1:
            return self._compute_multi_output_input_step(
                valid_output_heights,
                valid_output_widths,
                include_grad_safe=True,
            )

        valid_input_height = max(
            1,
            valid_output_heights[0] * int(self._output_stride_per_output[0][1]),
        )
        valid_input_width = max(
            1,
            valid_output_widths[0] * int(self._output_stride_per_output[0][2]),
        )
        grad_safe_h, grad_safe_w = self._compute_internal_safe_input_step()
        valid_input_height = min(valid_input_height, int(grad_safe_h))
        valid_input_width = min(valid_input_width, int(grad_safe_w))

        internal_align_h, internal_align_w = self._compute_internal_alignment()
        align_h = math.lcm(int(self._output_stride_per_output[0][1]), internal_align_h)
        align_w = math.lcm(int(self._output_stride_per_output[0][2]), internal_align_w)
        valid_input_height = max(align_h, (valid_input_height // align_h) * align_h)
        valid_input_width = max(align_w, (valid_input_width // align_w) * align_w)
        return valid_input_height, valid_input_width

    def _compute_tile_grid(
        self,
        image_height,
        image_width,
        tile_height,
        tile_width,
        valid_input_height,
        valid_input_width,
    ):
        """Compute tiling grid shape for a given image and tile step."""
        return tile_grid(image_height, image_width, tile_height, tile_width, valid_input_height, valid_input_width)

    def _iter_input_tiles(
        self,
        image,
        n_rows,
        n_cols,
        valid_input_height,
        valid_input_width,
        tile_height,
        tile_width,
    ):
        """Yield input-space tile coordinates with border-aware side markers."""
        yield from iter_tiles(
            image.shape[H_DIM], image.shape[W_DIM], tile_height, tile_width,
            valid_input_height, valid_input_width, n_rows, n_cols,
        )

    def _log_and_validate_tile_start(self, input_y, input_x, sides, internal_alignment):
        """Log tile starts and verify non-edge starts keep internal sampling phase."""
        align_h, align_w = internal_alignment
        logger.debug(
            "Forward tile start: y=%s, x=%s, sides=%s, internal_alignment=(%s, %s)",
            input_y,
            input_x,
            sides,
            align_h,
            align_w,
        )
        if not sides.bottom:
            assert input_y % align_h == 0, (
                f"Non-bottom-edge tile y-start {input_y} is not a multiple of " f"internal alignment {align_h}"
            )
        if not sides.right:
            assert input_x % align_w == 0, (
                f"Non-right-edge tile x-start {input_x} is not a multiple of " f"internal alignment {align_w}"
            )

    def _tile_start_list(self, tile_iter):
        """Return input-space tile starts for compact forward/backward diagnostics."""
        return [(int(input_y), int(input_x)) for input_y, input_x, _ in tile_iter]

    def _log_forward_tile_starts(self):
        logger.debug("forward tile starts: %s", self._tile_start_list(self._last_forward_tiles))

    def _log_backward_tile_starts(self, tile_iter):
        logger.debug("backward tile starts: %s", self._tile_start_list(tile_iter))

    def _validate_backward_tile_iter_matches_forward(self, tile_iter):
        """Assert in debug mode that backward replays the exact forward tile starts."""
        if not __debug__ or not self._last_forward_tiles:
            return

        forward_starts = self._tile_start_list(self._last_forward_tiles)
        backward_starts = self._tile_start_list(tile_iter)
        assert backward_starts == forward_starts, (
            "Backward tile starts differ from forward tile starts: "
            f"forward={forward_starts}, backward={backward_starts}"
        )






    def _trim_head_output(self, head_output, head_lost):
        return head_output[
            :,
            :,
            head_lost.top : head_output.shape[H_DIM] - head_lost.bottom,
            head_lost.left : head_output.shape[W_DIM] - head_lost.right,
        ]

    def _build_stitched_tile_output(self, head_idx, head_output, tile_input_y, tile_input_x, sides):
        head_lost = self._get_tile_lost_for_sides(sides, self._tile_output_lost[head_idx])
        head_stride = self._output_stride_per_output[head_idx]
        head_output_y = tile_input_y // int(head_stride[1])
        head_output_x = tile_input_x // int(head_stride[2])
        output_loc = Box(head_output_y + head_lost.top, -1, head_output_x + head_lost.left, -1, sides)
        trimmed_output = self._trim_head_output(head_output, head_lost)
        return head_lost, output_loc, trimmed_output

    def forward(self, image, result_on_cpu=False, mask=None):
        """Delegate a forward call to the plan-driven executor."""
        return self._forward_executor.execute(self.plan, ForwardCall(image, result_on_cpu, mask))


    def backward(self, image, grad, mask=None):
        """Delegate a backward call to the plan-driven executor."""
        return self._backward_executor.execute(self.plan, BackwardCall(image, grad, mask))




    def _get_tile_lost_for_sides(self, sides, output_lost=None):
        output_lost = self.tile_output_lost if output_lost is None else output_lost
        lost_top = output_lost.top if not sides.top else 0
        lost_bottom = output_lost.bottom if not sides.bottom else 0
        lost_left = output_lost.left if not sides.left else 0
        lost_right = output_lost.right if not sides.right else 0
        lost = Lost(lost_top, lost_left, lost_bottom, lost_right)
        return lost

    def _normalize_on_gpu(self, tile):
        tile_norm = tile.to(self.dtype)
        del tile
        tile_norm.div_(255)
        tile_norm.sub_(self.mean)
        tile_norm.div_(self.std)
        tile = tile_norm
        return tile

    def disable(self):
        """Disable the streaming hooks"""
        self._remove_hooks()
        self.stream_module = self._reset_converted_modules(self.stream_module)

    def enable(self):
        """Enable the streaming hooks"""
        self._remove_hooks()
        self._streaming_reducers = []
        self.stream_module = self._convert_modules_for_streaming(self.stream_module)
        self._add_hooks_for_streaming()

    def _add_hooks_for_statistics(self):
        def forw_lambda(module, inpt, outpt):
            self._forward_gather_statistics_hook(module, inpt, outpt)

        def back_lambda(module, grad_in, grad_out):
            return self._backward_gather_statistics_hook(module, grad_in, grad_out)

        self._add_hooks(forward_hook=forw_lambda, backward_hook=back_lambda)

    def _add_hooks_for_streaming(self):
        if self.gather_input_gradient:

            def back_lambda(module, grad_in, grad_out):
                return self._backward_saliency_hook(module, grad_in, grad_out)

            for mod in self.stream_module.modules():
                if isinstance(mod, StreamingConv2d):
                    if mod.in_channels == 3:
                        self.saliency_input_module = mod
                        back_handle = mod.register_full_backward_hook(back_lambda)
                        self._hooks.append(back_handle)

    def _add_hooks(
        self,
        forward_hook,
        backward_hook,
        forward_modules=(
            torch.nn.Conv2d,
            torch.nn.MaxPool2d,
            torch.nn.AvgPool2d,
            torch.nn.Upsample,
        ),
        back_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Upsample),
    ):
        for mod in self.stream_module.modules():
            register_forward = isinstance(mod, forward_modules) or _is_spatial_preserving_pointwise_module(mod)
            register_backward = back_modules and (isinstance(mod, back_modules) or _is_spatial_preserving_pointwise_module(mod))
            if register_forward:
                forw_handle = mod.register_forward_hook(forward_hook)
                self._hooks.append(forw_handle)
                if register_backward:
                    back_handle = mod.register_full_backward_hook(backward_hook)
                    self._hooks.append(back_handle)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def _resolve_upsample_scale(self, module, inpt, output):
        if module.scale_factor is not None:
            sf = module.scale_factor
            if isinstance(sf, tuple):
                return float(sf[-2]), float(sf[-1])
            return float(sf), float(sf)
        in_h, in_w = inpt[0].shape[H_DIM], inpt[0].shape[W_DIM]
        out_h, out_w = output.shape[H_DIM], output.shape[W_DIM]
        return float(out_h) / float(max(1, in_h)), float(out_w) / float(max(1, in_w))

    def _update_output_stride_for_upsample(self, prev_output_stride, scale_h, scale_w):
        out_stride = prev_output_stride.clone().detach().to(torch.float32)
        out_stride[1] = max(1.0, out_stride[1] / max(scale_h, 1e-8))
        out_stride[2] = max(1.0, out_stride[2] / max(scale_w, 1e-8))
        out_stride = torch.round(out_stride).to(torch.long)
        out_stride[0] = 1
        return out_stride

    def _forward_gather_statistics_hook(self, module, inpt, output):
        is_upsample = isinstance(module, torch.nn.Upsample)
        is_pointwise_module = _is_spatial_preserving_pointwise_module(module)
        is_merge = isinstance(module, StreamingMerge)
        if is_pointwise_module:
            stride = torch.tensor([1, 1, 1])
            kernel_size = torch.tensor([1, 1, 1])
            padding = torch.tensor([0, 0, 0])
            dilation = torch.tensor([1, 1, 1])
        elif not is_upsample:
            stride, kernel_size, padding, dilation = (
                _triple(module.stride),
                _triple(module.kernel_size),
                _triple(module.padding),
                _triple(getattr(module, "dilation", 1)),
            )
        else:
            stride = torch.tensor([1, 1, 1])
            kernel_size = torch.tensor([1, 1, 1])
            padding = torch.tensor([0, 0, 0])
            dilation = torch.tensor([1, 1, 1])

        if not torch.is_grad_enabled():  # type:ignore
            # Convert strided convolutions/pooling to average pool
            if (not is_upsample and not is_pointwise_module) and (
                isinstance(module, (torch.nn.MaxPool2d))
                or (stride[0] > 1 and stride[0] > kernel_size[0])
                or (stride[1] > 1 and stride[1] > kernel_size[1])
                or (stride[2] > 1 and stride[2] > kernel_size[2])
            ):
                # Pytorch documentation is explicitely against changing output in a forward hook
                # However, since we do not really need the graph or gradients to be correct
                # it shouldn't harm.
                if module.padding != 0:
                    padding = module.padding
                    if not isinstance(module.padding, tuple):
                        padding = [module.padding, module.padding]
                    padded_input = torch.nn.functional.pad(inpt[0], [padding[1], padding[1], padding[0], padding[0]])
                else:
                    padded_input = inpt[0]

                new_output = torch.nn.functional.avg_pool2d(padded_input, kernel_size[1:], stride[1:])
                new_output = torch.sum(new_output, dim=1)[0]
                new_output = new_output > (1 - self.eps) * new_output.max()
                new_output = new_output.expand_as(output[0])

                output[0] = new_output.type(self.dtype)

            # Sum all dimensions (useful for DenseNet-like networks). Channel-only
            # normalization and layer scaling preserve spatial support, but
            # constant setup tensors can make the actual output all zeros, so
            # derive validity from the input.

            # A merge's valid support is the intersection encoded by its actual
            # numerical result.  Other pointwise boundaries remain value
            # independent and inherit support from their sole input.
            validity_source = output if is_merge or not is_pointwise_module else inpt[0]
            lost = self._non_max_border_amount(validity_source)

            # Make output between 0-1 again, so the values do not explode
            output.fill_(0)
            output[
                :,
                :,
                lost.top : output[0, 0].shape[0] - lost.bottom,
                lost.left : output[0, 0].shape[1] - lost.right,
            ] = 1

            module_stats = {
                "lost": lost,
                "stride": stride if not is_upsample else torch.tensor([1, 1, 1]),
                "kernel_size": kernel_size,
                "padding": padding,
                "dilation": dilation,
                "module": module,
            }
            if is_upsample:
                module_stats["backward_valid_lost"] = Lost(0, 0, 0, 0)
                module_stats["upsample_forward_output_lost"] = module_stats["lost"]

            self._print_verbose(module, "\n", module_stats["lost"])

            self._saved_tensors[module] = inpt
            self._module_stats[module] = module_stats
        else:
            module_stats = self._module_stats[module]

            if is_merge:
                if inpt[0].shape[-2:] != inpt[1].shape[-2:]:
                    raise ValueError(
                        "StreamingMerge inputs must have compatible spatial shapes; "
                        f"got {tuple(inpt[0].shape[-2:])} and {tuple(inpt[1].shape[-2:])}"
                    )
                input_stats = [self._prev_stats(value) for value in inpt]
                coordinates = []
                for stats in input_stats:
                    if stats is None:
                        coordinates.append(((1, 1, 1), Lost(0, 0, 0, 0)))
                    else:
                        effective_stride = stats["output_stride"] * torch.as_tensor(stats["stride"])
                        coordinates.append(
                            (
                                tuple(int(value) for value in effective_stride),
                                stats["lost"],
                            )
                        )
                if coordinates[0] != coordinates[1]:
                    raise ValueError(
                        "StreamingMerge inputs must have compatible spatial coordinates; "
                        f"got {coordinates[0]} and {coordinates[1]}"
                    )

            p_stats = self._prev_stats(output)
            if p_stats:
                prev_output_stride = (
                    p_stats["output_stride"] * p_stats["stride"].clone().detach()
                    if isinstance(p_stats["stride"], torch.Tensor)
                    else p_stats["output_stride"] * torch.tensor(p_stats["stride"])
                )
            else:
                prev_output_stride = torch.tensor([1, 1, 1])

            if is_upsample:
                pre_upsample_output_stride = prev_output_stride.clone().detach()
                scale_h, scale_w = self._resolve_upsample_scale(module, inpt, output)
                output_stride = self._update_output_stride_for_upsample(prev_output_stride, scale_h, scale_w)
                pre_upsample_output_stride = output_stride.clone().detach().to(torch.float32)
                pre_upsample_output_stride[1] *= scale_h
                pre_upsample_output_stride[2] *= scale_w
                pre_upsample_output_stride = torch.round(pre_upsample_output_stride).to(torch.long)
                pre_upsample_output_stride[0] = 1
                module_stats["scale_factor_hw"] = (scale_h, scale_w)
                module_stats["pre_upsample_output_stride"] = pre_upsample_output_stride
            else:
                output_stride = prev_output_stride

            output_stride = output_stride.clone().detach()
            output_stride[0] = 1
            module_stats["output_stride"] = output_stride
            if is_upsample:
                module_stats["post_upsample_output_stride"] = output_stride
            self._stats_per_grad_fn[output.grad_fn] = module_stats
            self._module_stats[module] = module_stats

    def _bilinear_upsample_backward_input_valid_mask(self, grad_output, grad_input, grad_lost, scale_h, scale_w):
        """Return low-resolution grad-input cells with complete valid high-resolution support."""
        out_h, out_w = int(grad_output.shape[H_DIM]), int(grad_output.shape[W_DIM])
        in_h, in_w = int(grad_input.shape[H_DIM]), int(grad_input.shape[W_DIM])
        device = grad_output.device

        high_valid_y0 = int(grad_lost.top)
        high_valid_y1 = out_h - int(grad_lost.bottom)
        high_valid_x0 = int(grad_lost.left)
        high_valid_x1 = out_w - int(grad_lost.right)

        def contributing_outputs(input_idx, out_size, scale):
            outputs = []
            for out_idx in range(out_size):
                src = (float(out_idx) + 0.5) / float(scale) - 0.5
                lo = math.floor(src)
                hi = math.ceil(src)
                if input_idx == lo or input_idx == hi:
                    outputs.append(out_idx)
            return outputs

        y_valid = []
        for in_y in range(in_h):
            outputs = contributing_outputs(in_y, out_h, scale_h)
            y_valid.append(bool(outputs) and all(high_valid_y0 <= out_y < high_valid_y1 for out_y in outputs))

        x_valid = []
        for in_x in range(in_w):
            outputs = contributing_outputs(in_x, out_w, scale_w)
            x_valid.append(bool(outputs) and all(high_valid_x0 <= out_x < high_valid_x1 for out_x in outputs))

        valid_mask = torch.zeros((in_h, in_w), dtype=torch.bool, device=device)
        for in_y, valid_y in enumerate(y_valid):
            if not valid_y:
                continue
            for in_x, valid_x in enumerate(x_valid):
                if valid_x:
                    valid_mask[in_y, in_x] = True
        return valid_mask

    def _backward_gather_statistics_hook(self, module, grad_in, grad_out):
        is_upsample = isinstance(module, torch.nn.Upsample)
        is_pointwise_module = _is_spatial_preserving_pointwise_module(module)
        if is_pointwise_module:
            stride = torch.tensor([1, 1, 1])
            kernel_size = torch.tensor([1, 1, 1])
            _padding = torch.tensor([0, 0, 0])
            dilation = torch.tensor([1, 1, 1])
        elif not is_upsample:
            stride, kernel_size, _padding, dilation = (
                _triple(module.stride),
                _triple(module.kernel_size),
                _triple(module.padding),
                _triple(getattr(module, "dilation", 1)),
            )
        else:
            stride = torch.tensor([1, 1, 1])
            kernel_size = torch.tensor([1, 1, 1])
            _padding = torch.tensor([0, 0, 0])
            dilation = torch.tensor([1, 1, 1])
        dilation_h, dilation_w = dilation[1], dilation[2]
        kernel_h, kernel_w = kernel_size[1], kernel_size[2]
        effective_kernel_h = dilation_h * (kernel_h - 1) + 1
        effective_kernel_w = dilation_w * (kernel_w - 1) + 1
        if grad_in[0] is not None:
            # We sum over the channels to deal with networks that do different operations
            # on groups of channels. Channel-only normalization and layer scaling are pointwise in space,
            # so derive validity from grad_out instead of value-dependent grad_in.
            f_grad = torch.sum(grad_out[0] if is_pointwise_module else grad_in[0], dim=1)[0]
            if isinstance(module, (torch.nn.MaxPool2d)):
                # MaxPool shifts indices around, which break the calculation to
                # find valid gradient values. To fix this we do an average pool
                # with the same kernel-size and stride and repeat using the stride.

                # We have the input gradient for a max pool right now
                # The following computes the input gradient for an average pool instead
                # compute the output of the average pool, and multiply by output gradient

                # from forward statistics hook, modified input
                inpt = self._saved_tensors[module]
                padded_inpt = inpt[0]

                if module.padding != 0:
                    padded_inpt = torch.nn.functional.pad(
                        inpt[0],
                        [
                            module.padding,
                            module.padding,
                            module.padding,
                            module.padding,
                        ],
                        value=-1,
                    )

                new_outpt = torch.nn.functional.avg_pool2d(padded_inpt, kernel_size[1:], stride[1:])[0]
                new_outpt = torch.sum(new_outpt, dim=0)

                f_grad = torch.sum(grad_out[0], dim=1)[0]
                f_grad = f_grad * new_outpt
                f_grad = f_grad.cpu()
                f_grad = np.repeat(f_grad, stride[1], axis=0)
                f_grad = np.repeat(f_grad, stride[2], axis=1)
                grad = np.zeros(grad_in[0].shape[2:])

                self._print_verbose("testing shape gradient fix")
                grad[: f_grad.shape[0], : f_grad.shape[1]] = f_grad[: grad.shape[0], : grad.shape[1]]

                f_grad = torch.from_numpy(grad)
                f_grad = f_grad.to(self.device)

            if grad_out[0].numel() == 0 or torch.count_nonzero(grad_out[0]) == 0:
                # Some connected branches (e.g. zero-scaled passthrough links for graph connectivity)
                # produce valid but all-zero gradients during stats gathering; skip border inference.
                return grad_in

            grad_lost = self._non_max_border_amount(grad_out[0])

            self._print_verbose(module, "\n", grad_lost)
            self._module_stats[module]["grad_lost"] = grad_lost

            if is_upsample and module.mode == "bilinear" and getattr(module, "align_corners", None) in (None, False):
                scale_h, scale_w = self._resolve_upsample_scale(module, grad_in, grad_out[0])
                input_valid_mask = self._bilinear_upsample_backward_input_valid_mask(
                    grad_out[0], grad_in[0], grad_lost, scale_h, scale_w
                )
                input_lost_probe = input_valid_mask[None, None].expand(
                    grad_in[0].shape[B_DIM],
                    grad_in[0].shape[C_DIM],
                    *input_valid_mask.shape,
                )
                self._module_stats[module]["upsample_backward_input_lost"] = self._non_max_border_amount(
                    input_lost_probe.to(dtype=self.dtype) * 10 - 1
                )

            valid_grad = f_grad > (1 - self.eps) * f_grad.max()

            # When the effective kernel is larger than the stride we have some
            # _overlap_ of gradients, this overlap makes extra positions in the
            # input gradient invalid. Dilation increases the effective receptive
            # field, so use the effective kernel rather than the raw kernel size.
            # Pointwise spatial-preserving channel modules have no additional spatial loss.
            if (not is_upsample and not is_pointwise_module) and (
                (stride[1] > 1 and effective_kernel_h > stride[1])
                or (stride[2] > 1 and effective_kernel_w > stride[2])
            ):
                valid_lost = self._non_max_border_amount(f_grad)
                valid_grad.fill_(0)
                overlap_rows = effective_kernel_h - stride[1]
                overlap_cols = effective_kernel_w - stride[2]
                valid_grad[
                    valid_lost.top + overlap_rows : valid_grad.shape[0] - valid_lost.bottom - overlap_rows,
                    valid_lost.left + overlap_cols : valid_grad.shape[1] - valid_lost.right - overlap_cols,
                ] = 1

            new_grad_in = valid_grad[None].expand(grad_in[0].shape[1], *valid_grad.shape)[None]
            new_grad_in = new_grad_in.type(self.dtype) * 10 - 1
            new_grad_in_lost = self._non_max_border_amount(new_grad_in)
            self._module_stats[module]["backward_valid_lost"] = new_grad_in_lost
            self._module_stats[module]["side_aware_grad_lost"] = {
                "interior": grad_lost,
                "top": Lost(0, grad_lost.left, grad_lost.bottom, grad_lost.right),
                "bottom": Lost(grad_lost.top, grad_lost.left, 0, grad_lost.right),
                "left": Lost(grad_lost.top, 0, grad_lost.bottom, grad_lost.right),
                "right": Lost(grad_lost.top, grad_lost.left, grad_lost.bottom, 0),
            }

            return (new_grad_in, *grad_in[1:])

    def _backward_saliency_hook(
        self,
        module: StreamingConv2d,
        grad_in,
        grad_out,
        is_bias=False,
        change_grad=True,
    ):
        stride: List[int] = _triple(module.stride)  # type:ignore

        # Trim gradient of invalid values
        sides = module.input_loc.sides
        grad_lost = module.grad_lost  # type: Lost

        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0
        lost = Lost(lost_top, lost_left, lost_bottom, lost_right)

        grad = grad_out[0]
        valid_grad = grad[
            :,
            :,
            lost_top : grad.shape[H_DIM] - lost_bottom,
            lost_left : grad.shape[W_DIM] - lost_right,
        ]

        output_stride = module.output_stride * torch.tensor(stride)
        input_loc = module.input_loc

        # Move the location according to how many pixels have been trimmed
        # this will be the location of the valid gradient of this layer in relation
        # to the actual gradient in a normal backpass
        data_loc_y = int(input_loc.y // output_stride[1]) + lost_top
        data_loc_x = int(input_loc.x // output_stride[2]) + lost_left

        data_loc = Box(data_loc_y, 0, data_loc_x, 0, input_loc.sides)

        # Calculate which part of the gradient is 'new'
        old_value_indices = self.saliency_old_indices
        new_output_box, updated_total_indices = _new_value_indices(valid_grad.shape, data_loc, old_value_indices)

        if module.in_channels == 3:
            valid_grad_in = grad_in[0][
                :,
                :,
                lost.top * stride[1] : grad_in[0].shape[2] - lost.bottom * stride[1],
                lost.left * stride[2] : grad_in[0].shape[3] - lost.right * stride[2],
            ]

            relevant_input_grad = valid_grad_in[
                :,
                :,
                new_output_box.y * stride[1] : new_output_box.y * stride[1] + new_output_box.height * stride[1],
                new_output_box.x * stride[2] : new_output_box.x * stride[2] + new_output_box.width * stride[2],
            ]

            self.saliency_map[
                :,
                :,
                updated_total_indices.y * stride[1] : updated_total_indices.height * stride[1],
                updated_total_indices.x * stride[2]
                - relevant_input_grad.shape[3] : updated_total_indices.x * stride[2],
            ] = relevant_input_grad.detach().cpu()

            del relevant_input_grad
            del valid_grad_in
        return grad_in

    def _prev_stats(self, grad_fn):
        """DAG traversal, finds the first grad_fn that is in self._stats_per_grad_fn

        Finds the first grad_fn that is in self._stats_per_grad_fn, which is needed for output stride calculations

        Parameters
        ----------
        grad_fn: the grad function of the current output tensor

        """
        if hasattr(grad_fn, "grad_fn"):
            grad_fn = grad_fn.grad_fn

        prev_stats = None

        if grad_fn in self._stats_per_grad_fn:
            prev_stats = self._stats_per_grad_fn[grad_fn]
            return prev_stats
        elif hasattr(grad_fn, "next_functions") and len(grad_fn.next_functions) > 0:
            children = [x[0] for x in grad_fn.next_functions]

            for x in children:
                prev_stats = self._prev_stats(x)
                if prev_stats is not None:
                    break
            return prev_stats
        return prev_stats

    def get_tile_cache(self):
        named_stats = {"net_stats": {}}
        for name, module in self.stream_module.named_modules():
            if module in self._module_stats:
                named_stats["net_stats"][name] = self._module_stats[module]
        named_stats["output_stride"] = self.output_stride
        named_stats["tile_output_lost"] = self.tile_output_lost  # type:ignore
        named_stats["tile_output_lost_all"] = self._tile_output_lost  # type:ignore
        named_stats["tile_gradient_lost"] = self.tile_gradient_lost  # type:ignore
        named_stats["tile_output_shape"] = self._tile_output_shape  # type:ignore
        named_stats["tile_output_shapes"] = self._tile_output_shapes  # type:ignore
        named_stats["output_stride_per_output"] = self._output_stride_per_output  # type:ignore
        named_stats["output_spec"] = self._output_spec
        return named_stats

    def load_tile_cache(self, state):
        self.disable()

        self.output_stride = state["output_stride"]
        self.tile_output_lost = state["tile_output_lost"]
        self._tile_output_lost = state.get("tile_output_lost_all", [self.tile_output_lost])
        self.tile_gradient_lost = state["tile_gradient_lost"]
        self._tile_output_shape = state["tile_output_shape"]
        self._tile_output_shapes = state.get("tile_output_shapes", [self._tile_output_shape])
        self._output_stride_per_output = state.get("output_stride_per_output", [self.output_stride])
        self._base_output_stride = self._output_stride_per_output[0].clone()
        for stride in self._output_stride_per_output[1:]:
            self._base_output_stride[1] = min(int(self._base_output_stride[1]), int(stride[1]))
            self._base_output_stride[2] = min(int(self._base_output_stride[2]), int(stride[2]))
        self._output_spec = state.get("output_spec", ("tensor", None))

        for name, module in self.stream_module.named_modules():
            if name in state["net_stats"]:
                self._module_stats[module] = state["net_stats"][name]

        self.enable()

    def __call__(self, image, **kwargs):
        result_on_cpu = kwargs.pop("result_on_cpu", False)
        return self.forward(image, result_on_cpu=result_on_cpu, **kwargs)
