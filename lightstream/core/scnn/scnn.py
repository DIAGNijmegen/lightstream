"""
Author: Hans Pinckaers
MIT License
"""
import math
import copy
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.autograd
import torch.backends
import torch.nn.functional

from lightstream.core.scnn.utils import Sides, Box, Lost, _ntuple, _new_value_indices, B_DIM, C_DIM, H_DIM, W_DIM
from lightstream.core.scnn.streamingconv import StreamingConv2d
from lightstream.core.scnn.streamingupsample import StreamingUpsample2d
from lightstream.modules.reducer import Reducer, StreamingReducer


_triple = _ntuple(3)

@dataclass(frozen=True)
class ForwardContext:
    image: torch.Tensor
    tile_height: int
    tile_width: int
    output_heights: list
    output_widths: list
    valid_input_height: int
    valid_input_width: int
    n_rows: int
    n_cols: int
    result_device: torch.device


@dataclass(frozen=True)
class BackwardContext:
    image: torch.Tensor
    grad_tensors: list
    tile_height: int
    tile_width: int
    output_heights: list
    output_widths: list


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
            verbose (bool): will log various debugging relevant information (default is False)
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
        self._saved_tensors = {}
        self.debug_reducer_replay = False
        self._hooks = []
        self._last_forward_tiles = []
        self._streaming_reducers = []
        self._reducer_head_map = {}

        if state_dict is None:
            self._configure()
        else:
            self.load_tile_cache(state_dict)

    def _configure(self):
        # Save current model and cudnn flags, since we need to change them and restore later
        state_dict = self._save_parameters()
        (old_deterministic_flag, old_benchmark_flag) = self._set_cudnn_flags_to_determistic()
        self._reset_parameters_to_constant()

        # Add hooks to each layer to gather statistics
        self._add_hooks_for_statistics()
        self._set_reducer_passthrough(True)

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
        if self.verbose:
            print("")
        self._gather_backward_statistics(tile)

        # TODO; temp hack for tile sizes too big on gpu,
        if self.statistics_on_cpu:
            self.stream_module = self.stream_module.cuda()
            self.device = torch.device("cuda")  # type:ignore

        # Remove all hooks and add hooks for correcting gradients
        # during lightstream
        self._remove_hooks()
        self._set_reducer_passthrough(False)
        #
        self._restore_parameters(state_dict)
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

    def _set_reducer_passthrough(self, enabled: bool):
        for mod in self.stream_module.modules():
            if isinstance(mod, Reducer):
                mod._streaming_passthrough = enabled

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
        if self.verbose:
            print("\n", "Input gradient lost", self.tile_gradient_lost)

    def _gather_forward_statistics(self, tile):
        torch.set_grad_enabled(False)
        output = self.stream_module(tile)
        output_tensors, output_spec = self._flatten_output_structure(output)
        self._output_spec = output_spec
        self._tile_output_lost = [self._non_max_border_amount(out) for out in output_tensors]
        self.tile_output_lost = self._tile_output_lost[0]
        if self.verbose:
            print("\n", "Output lost", self._tile_output_lost)

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

    def _compute_internal_safe_input_step(self):
        """Compute conservative input-step bounds from per-layer backward stats."""
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

        # Fallback to global backward-safe span if per-layer stats are unavailable
        grad_safe_h = self.tile_shape[H_DIM] - self.tile_gradient_lost.top - self.tile_gradient_lost.bottom
        grad_safe_w = self.tile_shape[W_DIM] - self.tile_gradient_lost.left - self.tile_gradient_lost.right
        candidates_h.append(int(grad_safe_h))
        candidates_w.append(int(grad_safe_w))

        return max(1, min(candidates_h)), max(1, min(candidates_w))

    def _compute_internal_alignment(self):
        """Compute input-space alignment constraints from internal streamed layers.

        When output heads are upsampled back to stride-1, alignment based only on
        head output stride becomes 1 and can lose the internal phase constraints
        required by earlier strided conv layers.
        """

        align_h = 1
        align_w = 1
        for mod in self.stream_module.modules():
            if not isinstance(mod, StreamingConv2d):
                continue

            stride = _triple(mod.stride)
            output_stride = getattr(mod, "output_stride", torch.tensor([1, 1, 1]))
            eff_h = int(output_stride[1]) * int(stride[1])
            eff_w = int(output_stride[2]) * int(stride[2])
            align_h = math.lcm(align_h, max(1, eff_h))
            align_w = math.lcm(align_w, max(1, eff_w))

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
                mod.grad_lost = self._module_stats[module].get("grad_lost", Lost(0, 0, 0, 0))
                mod.output_stride = self._module_stats[module].get("output_stride", torch.tensor([1, 1, 1]))
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, Reducer):
            mod = StreamingReducer.from_reducer(module)
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
                stats["output_stride"] = module.output_stride
                self._module_stats[mod] = stats
            else:
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        elif isinstance(module, StreamingReducer):
            mod = module.to_reducer()
        for name, child in module.named_children():
            mod.add_module(name, self._reset_converted_modules(child))
        del module
        return mod


    def _validate_reducer_head_map_resolved(self):
        if not self._streaming_reducers:
            return

        resolved_reducers = set(self._reducer_head_map.values())
        unresolved = [reducer for reducer in self._streaming_reducers if reducer not in resolved_reducers]
        if unresolved:
            raise RuntimeError(
                "Reducer head mapping incomplete after forward tile sampling: "
                f"resolved={len(resolved_reducers)}, expected={len(self._streaming_reducers)}"
            )

    def _validate_reducer_lifecycle_for_backward(self):
        if not self._streaming_reducers:
            return
        if not self._reducer_head_map:
            raise RuntimeError(
                "Reducer backward replay requires prior streaming forward pass to resolve reducer heads."
            )

    def _resolve_reducer_head_map(self, flat_outputs):
        # Invariant: reducer-head resolution happens once per forward stream and remains stable
        # for the paired backward replay traversal.
        if self._reducer_head_map or not self._streaming_reducers:
            return

        output_id_to_index = {id(tensor): idx for idx, tensor in enumerate(flat_outputs)}
        for reducer in self._streaming_reducers:
            if reducer._last_output is None:
                continue
            output_index = output_id_to_index.get(id(reducer._last_output))
            if output_index is not None:
                self._reducer_head_map[output_index] = reducer

    def _reset_parameters_to_constant(self):
        for mod in self.stream_module.modules():
            if isinstance(mod, (torch.nn.Conv2d)):
                # to counter floating precision errors, we assign 1 to the weights and
                # normalize the output after the conv.
                torch.nn.init.constant_(mod.weight, 1)
                if mod.bias is not None:
                    torch.nn.init.constant_(mod.bias, 0)

        for m in self.stream_module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                # Perhaps change to torch.nn.init.ones_(m.weight) and zeros?
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

    def _compute_full_output_sizes(self, image):
        """Return per-head output sizes for the fully stitched image output."""
        output_heights = [
            (image.shape[H_DIM] - self.tile_shape[H_DIM]) // int(self._output_stride_per_output[idx][1]) + tile_shape[H_DIM]
            for idx, tile_shape in enumerate(self._tile_output_shapes)
        ]
        output_widths = [
            (image.shape[W_DIM] - self.tile_shape[W_DIM]) // int(self._output_stride_per_output[idx][2]) + tile_shape[W_DIM]
            for idx, tile_shape in enumerate(self._tile_output_shapes)
        ]
        return output_heights, output_widths

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
        return valid_input_height, valid_input_width

    def _compute_tile_grid(self, image_height, image_width, tile_height, tile_width, valid_input_height, valid_input_width):
        """Compute tiling grid shape for a given image and tile step."""
        n_rows = math.ceil(float(max(1, image_height - tile_height)) / float(valid_input_height)) + 1
        n_cols = math.ceil(float(max(1, image_width - tile_width)) / float(valid_input_width)) + 1

        if image_width <= tile_width:
            n_cols = 1
        if image_height <= tile_height:
            n_rows = 1
        return n_rows, n_cols

    def _iter_input_tiles(self, image, n_rows, n_cols, valid_input_height, valid_input_width, tile_height, tile_width):
        """Yield input-space tile coordinates with border-aware side markers."""
        for row in range(n_rows):
            for col in range(n_cols):
                input_y = row * valid_input_height
                input_x = col * valid_input_width

                sides_top = row == 0
                sides_left = col == 0
                sides_bottom = input_y + tile_height >= image.shape[H_DIM]
                sides_right = input_x + tile_width >= image.shape[W_DIM]
                sides = Sides(sides_left, sides_top, sides_right, sides_bottom)

                if sides_bottom:
                    input_y = max(image.shape[H_DIM] - tile_height, 0)
                if sides_right:
                    input_x = max(image.shape[W_DIM] - tile_width, 0)

                input_y = input_y if not sides.top else 0
                input_x = input_x if not sides.left else 0
                yield int(input_y), int(input_x), sides

    def _prepare_forward_outputs(self, image, output_heights, output_widths, result_device):
        outputs = [None] * len(self._tile_output_shapes)

        def allocate_non_reducer_outputs():
            for idx in range(len(self._tile_output_shapes)):
                if idx in self._reducer_head_map or outputs[idx] is not None:
                    continue
                outputs[idx] = torch.empty(
                    (image.shape[0], self._tile_output_shapes[idx][1], output_heights[idx], output_widths[idx]),
                    dtype=self.dtype,
                    device=result_device,
                ).fill_(999)

        return outputs, allocate_non_reducer_outputs

    def _run_forward_tile(self, image, input_y, input_x, tile_height, tile_width):
        tile = image[:, :, input_y : input_y + tile_height, input_x : input_x + tile_width]

        if not self.copy_to_gpu:
            tile = tile.to(self.device, non_blocking=True)

        if self.should_normalize:
            tile = self._normalize_on_gpu(tile)

        tile_output = self.stream_module(tile)
        tile_outputs, _ = self._flatten_output_structure(tile_output)
        return tile, tile_outputs

    def _stitch_non_reducer_output(self, outputs, idx, trimmed_output, output_loc):
        src_y0 = 0
        src_y1 = int(trimmed_output.shape[H_DIM])
        src_x0 = 0
        src_x1 = int(trimmed_output.shape[W_DIM])

        dst_y0 = int(output_loc.y)
        dst_y1 = int(output_loc.y + trimmed_output.shape[H_DIM])
        dst_x0 = int(output_loc.x)
        dst_x1 = int(output_loc.x + trimmed_output.shape[W_DIM])

        clip_top = max(0, -dst_y0)
        clip_left = max(0, -dst_x0)
        clip_bottom = max(0, dst_y1 - outputs[idx].shape[H_DIM])
        clip_right = max(0, dst_x1 - outputs[idx].shape[W_DIM])

        if clip_top or clip_left or clip_bottom or clip_right:
            src_y0 += clip_top
            src_x0 += clip_left
            src_y1 -= clip_bottom
            src_x1 -= clip_right
            dst_y0 += clip_top
            dst_x0 += clip_left
            dst_y1 -= clip_bottom
            dst_x1 -= clip_right

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            return

        assert (dst_y1 - dst_y0) == trimmed_output[:, :, src_y0:src_y1, src_x0:src_x1].shape[H_DIM], (
            f"Y-shape mismatch while stitching output head {idx}: "
            f"dst=({dst_y0}:{dst_y1}) src_h={trimmed_output[:, :, src_y0:src_y1, src_x0:src_x1].shape[H_DIM]}"
        )
        assert (dst_x1 - dst_x0) == trimmed_output[:, :, src_y0:src_y1, src_x0:src_x1].shape[W_DIM], (
            f"X-shape mismatch while stitching output head {idx}: "
            f"dst=({dst_x0}:{dst_x1}) src_w={trimmed_output[:, :, src_y0:src_y1, src_x0:src_x1].shape[W_DIM]}"
        )

        outputs[idx][:, :, dst_y0:dst_y1, dst_x0:dst_x1] = trimmed_output[:, :, src_y0:src_y1, src_x0:src_x1]

    def _build_head_output_window(self, head_idx, tile_input_y, tile_input_x, sides, output_heights, output_widths, head_grad):
        head_stride = self._output_stride_per_output[head_idx]
        head_tile_height = self._tile_output_shapes[head_idx][H_DIM]
        head_tile_width = self._tile_output_shapes[head_idx][W_DIM]
        is_reducer_head = head_idx in self._reducer_head_map

        head_output_y = tile_input_y // int(head_stride[1])
        head_output_x = tile_input_x // int(head_stride[2])

        if sides.bottom:
            if is_reducer_head:
                head_output_y = max(output_heights[head_idx] - head_tile_height, 0)
            else:
                head_output_y = max(head_grad.shape[H_DIM] - head_tile_height, 0)
        if sides.right:
            if is_reducer_head:
                head_output_x = max(output_widths[head_idx] - head_tile_width, 0)
            else:
                head_output_x = max(head_grad.shape[W_DIM] - head_tile_width, 0)

        return int(head_output_y), int(head_output_x), bool(is_reducer_head)

    def _slice_non_reducer_gradient(self, head_grad, head_output_y, head_output_x, head_tile_height, head_tile_width):
        return head_grad[
            :,
            :,
            head_output_y : head_output_y + head_tile_height,
            head_output_x : head_output_x + head_tile_width,
        ]

    def _build_reducer_gradient(self, head_grad):
        gradient = head_grad.to(self.device, non_blocking=True)
        if gradient.shape[H_DIM] != 1 or gradient.shape[W_DIM] != 1:
            raise ValueError(f"Reducer-backed head expects gradient of shape N,C,1,1, got {tuple(gradient.shape)}")
        return gradient

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

    def _accumulate_reducer_forward_tile(self, head_idx, trimmed_output, output_loc, tile_input_y, tile_input_x, sides):
        dst_y0 = int(output_loc.y)
        dst_y1 = int(output_loc.y + trimmed_output.shape[H_DIM])
        dst_x0 = int(output_loc.x)
        dst_x1 = int(output_loc.x + trimmed_output.shape[W_DIM])
        reducer_mask_tile = None
        if self._reducer_mask is not None:
            reducer_mask_tile = self._reducer_mask[head_idx][dst_y0:dst_y1, dst_x0:dst_x1]
        self._reducer_head_map[head_idx].accumulate_stream_tile(
            trimmed_output=trimmed_output,
            tile_y=int(tile_input_y),
            tile_x=int(tile_input_x),
            sides=sides,
            dst_box=(dst_y0, dst_y1, dst_x0, dst_x1),
            mask_tile=reducer_mask_tile,
        )

    def _prepare_reducer_mask(self, mask, output_heights, output_widths):
        if mask is None:
            return None
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D (H,W), got shape={tuple(mask.shape)}")

        reducer_masks = {}
        for head_idx in self._reducer_head_map:
            expected_h = int(output_heights[head_idx])
            expected_w = int(output_widths[head_idx])
            if tuple(mask.shape) != (expected_h, expected_w):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match reducer head {head_idx} output shape {(expected_h, expected_w)}"
                )
            reducer_masks[head_idx] = mask.to(device=self.device, dtype=torch.bool)
        return reducer_masks

    def _stitch_forward_outputs(self, outputs, tile_outputs, tile_input_y, tile_input_x, sides):
        for head_idx, head_output in enumerate(tile_outputs):
            _, output_loc, trimmed_output = self._build_stitched_tile_output(
                head_idx=head_idx,
                head_output=head_output,
                tile_input_y=tile_input_y,
                tile_input_x=tile_input_x,
                sides=sides,
            )

            if head_idx in self._reducer_head_map:
                self._accumulate_reducer_forward_tile(
                    head_idx=head_idx,
                    trimmed_output=trimmed_output,
                    output_loc=output_loc,
                    tile_input_y=tile_input_y,
                    tile_input_x=tile_input_x,
                    sides=sides,
                )
                continue

            self._stitch_non_reducer_output(outputs, head_idx, trimmed_output, output_loc)

    def _prepare_backward_tile_iter_single_head(self, image, grad_tensors, tile_height, tile_width):
        grad_lost = self.tile_gradient_lost
        output_height = self._tile_output_shape[H_DIM]
        output_width = self._tile_output_shape[W_DIM]
        valid_grad_height = (tile_height - grad_lost.top - grad_lost.bottom) // int(self.output_stride[1])
        valid_grad_height *= int(self.output_stride[1])
        valid_grad_width = (tile_width - grad_lost.left - grad_lost.right) // int(self.output_stride[2])
        valid_grad_width *= int(self.output_stride[2])

        n_rows = math.ceil(float(image.shape[H_DIM] - grad_lost.top - grad_lost.bottom) / float(valid_grad_height))
        n_cols = math.ceil(float(image.shape[W_DIM] - grad_lost.left - grad_lost.right) / float(valid_grad_width))

        if image.shape[W_DIM] <= tile_width:
            n_cols = 1
        if image.shape[H_DIM] <= tile_height:
            n_rows = 1

        base_grad = grad_tensors[0]
        tile_iter = []
        for row in range(n_rows):
            for col in range(n_cols):
                output_y = row * valid_grad_height // int(self.output_stride[1])
                output_x = col * valid_grad_width // int(self.output_stride[2])

                sides_top = row == 0
                sides_left = col == 0
                sides_bottom = output_y + output_height >= base_grad.shape[H_DIM]
                sides_right = output_x + output_width >= base_grad.shape[W_DIM]

                if sides_bottom:
                    output_y = max(base_grad.shape[H_DIM] - output_height, 0)
                if sides_right:
                    output_x = max(base_grad.shape[W_DIM] - output_width, 0)

                input_y = output_y * int(self.output_stride[1])
                input_x = output_x * int(self.output_stride[2])
                tile_iter.append((int(input_y), int(input_x), Sides(sides_left, sides_top, sides_right, sides_bottom)))

        return tile_iter

    def _prepare_backward_tile_iter_multi_head(self, image, n_rows, n_cols, valid_input_height, valid_input_width, tile_height, tile_width):
        return list(
            self._iter_input_tiles(
                image=image,
                n_rows=n_rows,
                n_cols=n_cols,
                valid_input_height=valid_input_height,
                valid_input_width=valid_input_width,
                tile_height=tile_height,
                tile_width=tile_width,
            )
        )

    def _run_backward_tile(self, backward_ctx, input_y, input_x, sides):
        input_loc = Box(input_y, backward_ctx.tile_height, input_x, backward_ctx.tile_width, sides)
        tile = backward_ctx.image[:, :, input_y : input_y + backward_ctx.tile_height, input_x : input_x + backward_ctx.tile_width]

        self._saved_tensors = {}

        if not self.copy_to_gpu:
            tile = tile.to(self.device, non_blocking=True)

        for mod in self.stream_module.modules():
            if isinstance(mod, (StreamingConv2d, StreamingUpsample2d)):
                mod.input_loc = input_loc

        if self.should_normalize:
            tile = self._normalize_on_gpu(tile)

        if self.gather_input_gradient:
            tile.requires_grad = True
            self.saliency_old_indices = copy.deepcopy(self.saliency_input_module.seen_indices)

        use_cuda_autocast = self.device.type == "cuda" and torch.cuda.is_available()
        if use_cuda_autocast:
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                tile_output = self.stream_module(tile)
        else:
            tile_output = self.stream_module(tile)
        tile_outputs, _ = self._flatten_output_structure(tile_output)

        del tile

        trimmed_outputs = []
        trimmed_grads = []
        for idx, (head_output, head_grad) in enumerate(zip(tile_outputs, backward_ctx.grad_tensors)):
            paired_output, paired_grad = self._build_head_backward_pair(
                head_idx=idx,
                head_output=head_output,
                head_grad=head_grad,
                tile_input_y=input_y,
                tile_input_x=input_x,
                sides=sides,
                backward_ctx=backward_ctx,
            )
            trimmed_outputs.append(paired_output)
            trimmed_grads.append(paired_grad)

        torch.autograd.backward(trimmed_outputs, trimmed_grads)

        del tile_output
        del trimmed_grads
        del trimmed_outputs

    def forward(self, image, result_on_cpu=False, mask=None):
        """Perform forward pass with lightstream."""
        if self.copy_to_gpu:
            image = image.to(self.device, non_blocking=True)

        tile_height = self.tile_shape[H_DIM]
        tile_width = self.tile_shape[W_DIM]

        valid_output_heights, valid_output_widths = self._compute_valid_output_sizes()
        output_heights, output_widths = self._compute_full_output_sizes(image)
        valid_input_height, valid_input_width = self._compute_valid_input_step(valid_output_heights, valid_output_widths)
        n_rows, n_cols = self._compute_tile_grid(
            image_height=image.shape[H_DIM],
            image_width=image.shape[W_DIM],
            tile_height=tile_height,
            tile_width=tile_width,
            valid_input_height=valid_input_height,
            valid_input_width=valid_input_width,
        )

        if self.gather_input_gradient:
            self.saliency_map = torch.zeros(image.shape, dtype=self.dtype, device=self.device)

        self._last_forward_tiles = []
        if self.verbose:
            print(
                f"Forward tiling step: valid_input_height={valid_input_height}, "
                f"valid_input_width={valid_input_width}, tiles={n_rows}x{n_cols}={n_rows * n_cols}"
            )

        result_device = torch.device("cpu") if result_on_cpu else self.device
        forward_ctx = ForwardContext(
            image=image,
            tile_height=tile_height,
            tile_width=tile_width,
            output_heights=output_heights,
            output_widths=output_widths,
            valid_input_height=valid_input_height,
            valid_input_width=valid_input_width,
            n_rows=n_rows,
            n_cols=n_cols,
            result_device=result_device,
        )
        self._reducer_head_map = {}
        self._reducer_mask = None
        reducers_initialized = False
        outputs, allocate_non_reducer_outputs = self._prepare_forward_outputs(
            image=forward_ctx.image,
            output_heights=forward_ctx.output_heights,
            output_widths=forward_ctx.output_widths,
            result_device=forward_ctx.result_device,
        )

        last_sides = None
        with torch.no_grad():
            for input_y, input_x, sides in self._iter_input_tiles(
                image=forward_ctx.image,
                n_rows=forward_ctx.n_rows,
                n_cols=forward_ctx.n_cols,
                valid_input_height=forward_ctx.valid_input_height,
                valid_input_width=forward_ctx.valid_input_width,
                tile_height=forward_ctx.tile_height,
                tile_width=forward_ctx.tile_width,
            ):
                last_sides = sides
                self._last_forward_tiles.append((input_y, input_x, sides))
                tile, tile_outputs = self._run_forward_tile(forward_ctx.image, input_y, input_x, forward_ctx.tile_height, forward_ctx.tile_width)

                self._resolve_reducer_head_map(tile_outputs)
                allocate_non_reducer_outputs()

                if self._reducer_head_map and not reducers_initialized:
                    self._reducer_mask = self._prepare_reducer_mask(mask, forward_ctx.output_heights, forward_ctx.output_widths)
                    for head_idx, reducer in self._reducer_head_map.items():
                        reducer.start_stream(
                            output_height=forward_ctx.output_heights[head_idx],
                            output_width=forward_ctx.output_widths[head_idx],
                            batch_size=forward_ctx.image.shape[B_DIM],
                            channels=self._tile_output_shapes[head_idx][C_DIM],
                            device=self.device,
                            dtype=self.dtype,
                            debug_replay=self.debug_reducer_replay,
                            user_mask=None if self._reducer_mask is None else self._reducer_mask[head_idx],
                        )
                    reducers_initialized = True

                if torch.backends.cudnn.benchmark:
                    torch.cuda.empty_cache()

                self._stitch_forward_outputs(outputs, tile_outputs, input_y, input_x, sides)
                del tile

        assert last_sides is not None and last_sides.bottom and last_sides.right, (
            "It seems like we could not reconstruct all output"
        )

        self._validate_reducer_head_map_resolved()

        del image
        self._saved_tensors = {}
        for idx, reducer in self._reducer_head_map.items():
            outputs[idx] = reducer.finish_stream().to(result_device)

        for idx, output in enumerate(outputs):
            if output is None:
                raise RuntimeError(f"Output head {idx} was not populated during streaming forward.")

        output, final_idx = self._unflatten_output_structure(outputs, self._output_spec)
        assert final_idx == len(outputs)
        return output

    def backward(self, image, grad):
        """Perform backward pass with lightstream."""
        if self.copy_to_gpu:
            image = image.to(self.device, non_blocking=True)

        tile_height = self.tile_shape[H_DIM]
        tile_width = self.tile_shape[W_DIM]

        valid_output_heights, valid_output_widths = self._compute_valid_output_sizes()
        output_heights, output_widths = self._compute_full_output_sizes(image)
        valid_input_height, valid_input_width = self._compute_valid_input_step(valid_output_heights, valid_output_widths)
        n_rows, n_cols = self._compute_tile_grid(
            image_height=image.shape[H_DIM],
            image_width=image.shape[W_DIM],
            tile_height=tile_height,
            tile_width=tile_width,
            valid_input_height=valid_input_height,
            valid_input_width=valid_input_width,
        )

        grad_tensors, grad_spec = self._flatten_output_structure(grad)
        if grad_spec != self._output_spec:
            raise ValueError("Gradient output structure does not match streaming output structure")

        if len(self._tile_output_shapes) == 1:
            tile_iter = self._prepare_backward_tile_iter_single_head(image, grad_tensors, tile_height, tile_width)
        else:
            tile_iter = self._prepare_backward_tile_iter_multi_head(
                image=image,
                n_rows=n_rows,
                n_cols=n_cols,
                valid_input_height=valid_input_height,
                valid_input_width=valid_input_width,
                tile_height=tile_height,
                tile_width=tile_width,
            )

        self._validate_reducer_lifecycle_for_backward()

        backward_ctx = BackwardContext(
            image=image,
            grad_tensors=grad_tensors,
            tile_height=tile_height,
            tile_width=tile_width,
            output_heights=output_heights,
            output_widths=output_widths,
        )

        for reducer in self._reducer_head_map.values():
            reducer.start_backward_replay()

        last_sides = None
        for input_y, input_x, sides in tile_iter:
            last_sides = sides
            self._run_backward_tile(
                backward_ctx=backward_ctx,
                input_y=input_y,
                input_x=input_x,
                sides=sides,
            )

        if self.debug_reducer_replay:
            for idx, reducer in self._reducer_head_map.items():
                reducer.validate_backward_replay_consumed(head_idx=idx)

        self._saved_tensors = {}

        for mod in self.stream_module.modules():
            if isinstance(mod, (StreamingConv2d, StreamingUpsample2d)):
                mod.input_loc = None
                mod.reset()

        assert last_sides is not None and last_sides.right and last_sides.bottom, (
            "It seems like we could not reconstruct all output"
        )

    def _build_head_backward_pair(
        self,
        head_idx,
        head_output,
        head_grad,
        tile_input_y,
        tile_input_x,
        sides,
        backward_ctx,
    ):
        head_lost = self._get_tile_lost_for_sides(sides, self._tile_output_lost[head_idx])
        head_tile_height = self._tile_output_shapes[head_idx][H_DIM]
        head_tile_width = self._tile_output_shapes[head_idx][W_DIM]
        head_output_y, head_output_x, is_reducer_head = self._build_head_output_window(
            head_idx=head_idx,
            tile_input_y=tile_input_y,
            tile_input_x=tile_input_x,
            sides=sides,
            output_heights=backward_ctx.output_heights,
            output_widths=backward_ctx.output_widths,
            head_grad=head_grad,
        )

        if is_reducer_head:
            gradient = self._build_reducer_gradient(head_grad)
        else:
            gradient = self._slice_non_reducer_gradient(
                head_grad=head_grad,
                head_output_y=head_output_y,
                head_output_x=head_output_x,
                head_tile_height=head_tile_height,
                head_tile_width=head_tile_width,
            )

        trimmed_output = self._trim_head_output(head_output, head_lost)
        trimmed_output = trimmed_output.to(self.device, non_blocking=True)

        if is_reducer_head:
            return self._build_reducer_backward_pair(
                head_idx=head_idx,
                trimmed_output=trimmed_output,
                gradient=gradient,
                head_output_y=head_output_y,
                head_output_x=head_output_x,
                tile_input_y=tile_input_y,
                tile_input_x=tile_input_x,
                sides=sides,
            )

        return self._build_non_reducer_backward_pair(
            trimmed_output=trimmed_output,
            gradient=gradient,
            head_lost=head_lost,
            image=backward_ctx.image,
        )

    def _build_non_reducer_backward_pair(self, trimmed_output, gradient, head_lost, image):
        trimmed_grad = gradient[
            :,
            :,
            head_lost.top : gradient.shape[H_DIM] - head_lost.bottom,
            head_lost.left : gradient.shape[W_DIM] - head_lost.right,
        ]

        if trimmed_grad.shape[H_DIM] != trimmed_output.shape[H_DIM] or trimmed_grad.shape[W_DIM] != trimmed_output.shape[W_DIM]:
            assert image.shape[H_DIM] < self.tile_shape[H_DIM] or image.shape[W_DIM] < self.tile_shape[W_DIM]
            trimmed_grad = trimmed_grad[:, :, 0 : trimmed_output.shape[H_DIM], 0 : trimmed_output.shape[W_DIM]]

        return trimmed_output, trimmed_grad

    def _build_reducer_backward_pair(
        self,
        head_idx,
        trimmed_output,
        gradient,
        head_output_y,
        head_output_x,
        tile_input_y,
        tile_input_x,
        sides,
    ):
        reducer = self._reducer_head_map[head_idx]
        dst_box = (
            int(head_output_y),
            int(head_output_y + trimmed_output.shape[H_DIM]),
            int(head_output_x),
            int(head_output_x + trimmed_output.shape[W_DIM]),
        )

        reduced_output, reduced_grad = reducer.build_backward_pair(
            trimmed_output,
            gradient,
            input_y=int(tile_input_y),
            input_x=int(tile_input_x),
            sides=sides,
            dst_box=dst_box,
        )

        return reduced_output, reduced_grad

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
        forward_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.Upsample),
        back_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Upsample),
    ):
        for mod in self.stream_module.modules():
            if isinstance(mod, forward_modules):
                forw_handle = mod.register_forward_hook(forward_hook)
                self._hooks.append(forw_handle)
                if back_modules and isinstance(mod, back_modules):
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
        if not is_upsample:
            stride, kernel_size, _ = (_triple(module.stride), _triple(module.kernel_size), _triple(module.padding))
        else:
            stride = torch.tensor([1, 1, 1])
            kernel_size = torch.tensor([1, 1, 1])

        if not torch.is_grad_enabled():  # type:ignore
            # Convert strided convolutions/pooling to average pool
            if (not is_upsample) and (
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

            # Sum all dimensions (useful for DenseNet like networks)
            lost = self._non_max_border_amount(output)

            # Make output between 0-1 again, so the values do not explode
            output.fill_(0)
            output[
                :, :, lost.top : output[0, 0].shape[0] - lost.bottom, lost.left : output[0, 0].shape[1] - lost.right
            ] = 1

            module_stats = {"lost": lost, "stride": stride if not is_upsample else torch.tensor([1, 1, 1]), "module": module}
            if self.verbose:
                print(module, "\n", module_stats["lost"])

            self._saved_tensors[module] = inpt
            self._module_stats[module] = module_stats
        else:
            module_stats = self._module_stats[module]

            p_stats = self._prev_stats(output)
            if p_stats:
                prev_output_stride = p_stats["output_stride"] * p_stats["stride"].clone().detach() if isinstance(p_stats["stride"], torch.Tensor) else p_stats["output_stride"] * torch.tensor(p_stats["stride"])
            else:
                prev_output_stride = torch.tensor([1, 1, 1])

            if is_upsample:
                scale_h, scale_w = self._resolve_upsample_scale(module, inpt, output)
                output_stride = self._update_output_stride_for_upsample(prev_output_stride, scale_h, scale_w)
                module_stats["scale_factor_hw"] = (scale_h, scale_w)
            else:
                output_stride = prev_output_stride

            module_stats["output_stride"] = output_stride.clone().detach()
            self._stats_per_grad_fn[output.grad_fn] = module_stats
            self._module_stats[module] = module_stats

    def _backward_gather_statistics_hook(self, module, grad_in, grad_out):
        is_upsample = isinstance(module, torch.nn.Upsample)
        if not is_upsample:
            stride, kernel_size, _ = (_triple(module.stride), _triple(module.kernel_size), _triple(module.padding))
        if grad_in[0] is not None:
            # We sum over the channels to deal with networks that do different operations
            # on groups of channels
            f_grad = torch.sum(grad_in[0], dim=1)[0]
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
                        inpt[0], [module.padding, module.padding, module.padding, module.padding], value=-1
                    )

                new_outpt = torch.nn.functional.avg_pool2d(padded_inpt, kernel_size[1:], stride[1:])[0]
                new_outpt = torch.sum(new_outpt, dim=0)

                f_grad = torch.sum(grad_out[0], dim=1)[0]
                f_grad = f_grad * new_outpt
                f_grad = f_grad.cpu()
                f_grad = np.repeat(f_grad, stride[1], axis=0)
                f_grad = np.repeat(f_grad, stride[2], axis=1)
                grad = np.zeros(grad_in[0].shape[2:])

                print("testing shape gradient fix")
                grad[: f_grad.shape[0], : f_grad.shape[1]] = f_grad[: grad.shape[0], : grad.shape[1]]

                f_grad = torch.from_numpy(grad)
                f_grad = f_grad.to(self.device)

            grad_lost = self._non_max_border_amount(grad_out[0])

            if self.verbose:
                print(module, "\n", grad_lost)
            self._module_stats[module]["grad_lost"] = grad_lost

            valid_grad = f_grad > (1 - self.eps) * f_grad.max()

            # When kernel_size > stride we have some _overlap_ of gradients,
            # this overlap makes extra positions in the input gradient invalid
            if (not is_upsample) and (
                (stride[0] > 1 and kernel_size[0] > stride[0])
                or (stride[1] > 1 and kernel_size[1] > stride[1])
                or (stride[2] > 1 and kernel_size[2] > stride[2])
            ):
                valid_lost = self._non_max_border_amount(f_grad)
                valid_grad.fill_(0)
                overlap_rows = kernel_size[1] - stride[1]
                overlap_cols = kernel_size[2] - stride[2]
                valid_grad[
                    valid_lost.top + overlap_rows : valid_grad.shape[0] - valid_lost.bottom - overlap_rows,
                    valid_lost.left + overlap_cols : valid_grad.shape[1] - valid_lost.right - overlap_cols,
                ] = 1

            new_grad_in = valid_grad[None].expand(grad_in[0].shape[1], *valid_grad.shape)[None]
            new_grad_in = new_grad_in.type(self.dtype) * 10 - 1
            new_grad_in_lost = self._non_max_border_amount(new_grad_in)

            return (new_grad_in, *grad_in[1:])

    def _backward_saliency_hook(self, module: StreamingConv2d, grad_in, grad_out, is_bias=False, change_grad=True):
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
        valid_grad = grad[:, :, lost_top : grad.shape[H_DIM] - lost_bottom, lost_left : grad.shape[W_DIM] - lost_right]

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
        return self.forward(image, result_on_cpu)
