"""Tile execution for the plan-driven streaming engine.

The executors own traversal and tensor plumbing.  Reducer coordination remains
behind the small set of ``runtime._*reducer*`` calls until its separate
extraction; model/statistics policy deliberately stays on the runtime.
"""

import copy
from dataclasses import dataclass
from typing import Any

import torch

from .configuration import StreamingPlan
from .geometry import B_DIM, C_DIM, H_DIM, W_DIM, Box, full_output_sizes, iter_tiles
from .stitching import stitch_clipped


@dataclass(frozen=True)
class ForwardCall:
    image: torch.Tensor
    result_on_cpu: bool = False
    mask: torch.Tensor | None = None


@dataclass(frozen=True)
class BackwardCall:
    image: torch.Tensor
    gradient: Any
    mask: torch.Tensor | None = None


@dataclass(frozen=True)
class ForwardExecutionContext:
    image: torch.Tensor
    tile_height: int
    tile_width: int
    output_heights: list[int]
    output_widths: list[int]
    step_height: int
    step_width: int
    rows: int
    cols: int
    result_device: torch.device


@dataclass(frozen=True)
class BackwardExecutionContext:
    image: torch.Tensor
    gradients: list[Any]
    tile_height: int
    tile_width: int
    output_heights: list[int]
    output_widths: list[int]


class ForwardExecutor:
    def __init__(self, runtime):
        self.runtime = runtime

    def execute(self, plan: StreamingPlan, call: ForwardCall):
        r = self.runtime
        if plan is not r.plan:
            raise ValueError("ForwardExecutor received a plan for a different runtime")
        image = call.image.to(r.device, non_blocking=True) if r.copy_to_gpu else call.image
        r._active_reducer_mask, r._active_reducer_mask_image = call.mask, image
        r._prepared_reducer_domain_masks = {}
        th, tw = r.tile_shape[H_DIM], r.tile_shape[W_DIM]
        valid_h, valid_w = r._compute_valid_output_sizes()
        output_h, output_w = full_output_sizes(image.shape[H_DIM], image.shape[W_DIM], th, tw,
                                               r._tile_output_shapes, r._output_stride_per_output)
        r._current_output_heights, r._current_output_widths = output_h, output_w
        step_h, step_w = r._compute_valid_input_step(valid_h, valid_w)
        rows, cols = r._compute_tile_grid(image.shape[H_DIM], image.shape[W_DIM], th, tw, step_h, step_w)
        ctx = ForwardExecutionContext(image, th, tw, output_h, output_w, step_h, step_w, rows, cols,
                                      torch.device("cpu") if call.result_on_cpu else r.device)
        if r.gather_input_gradient:
            r.saliency_map = torch.zeros(image.shape, dtype=r.dtype, device="cpu")
        r._last_forward_tiles = []
        r._reducer_head_map, r._reducer_input_indices = {}, {}
        outputs = [None] * len(r._tile_output_shapes)
        reducers_initialized = False
        last_sides = None
        with torch.no_grad():
            for y, x, sides in iter_tiles(image.shape[H_DIM], image.shape[W_DIM], th, tw,
                                          step_h, step_w, rows, cols):
                last_sides = sides
                r._last_forward_tiles.append((y, x, sides))
                r._log_and_validate_tile_start(y, x, sides, r._compute_internal_alignment())
                tile = image[:, :, y:y + th, x:x + tw]
                if not r.copy_to_gpu:
                    tile = tile.to(r.device, non_blocking=True)
                if r.should_normalize:
                    tile = r._normalize_on_gpu(tile)
                tile_value = r.stream_module(tile)
                tile_outputs, _ = r._flatten_output_structure(tile_value)
                r._resolve_reducer_head_map(tile_outputs)
                self._allocate_outputs(outputs, ctx)
                if r._reducer_head_map and not reducers_initialized:
                    self._start_reducers(ctx)
                    reducers_initialized = True
                if torch.backends.cudnn.benchmark:
                    torch.cuda.empty_cache()
                self._stitch(outputs, tile_outputs, y, x, sides, call.mask)
        assert last_sides and last_sides.bottom and last_sides.right, "It seems like we could not reconstruct all output"
        r._log_forward_tile_starts()
        r._validate_reducer_head_map_resolved()
        r._saved_tensors = {}
        for idx, reducer in r._reducer_head_map.items():
            outputs[idx] = reducer.finish_stream().to(ctx.result_device)
        public = r._public_output_indices()
        r._validate_public_output_indices(public)
        expected = r._count_tensors_in_spec(r._output_spec)
        if len(public) != expected:
            raise RuntimeError(f"Public output index count mismatch: expected={expected}, actual={len(public)}; "
                               f"{r._public_output_debug_context(public)}")
        r._validate_public_forward_outputs(outputs, public)
        materialized = [outputs[i] for i in public]
        result, final = r._unflatten_output_structure(materialized, r._output_spec)
        assert final == len(materialized)
        return result

    def _allocate_outputs(self, outputs, ctx):
        r = self.runtime
        auxiliary = r._reducer_aux_indices()
        for idx, shape in enumerate(r._tile_output_shapes):
            if idx in r._reducer_head_map or idx in auxiliary or outputs[idx] is not None:
                continue
            outputs[idx] = torch.full((ctx.image.shape[B_DIM], shape[C_DIM], ctx.output_heights[idx],
                                       ctx.output_widths[idx]), 999, dtype=r.dtype, device=ctx.result_device)

    def _start_reducers(self, ctx):
        r = self.runtime
        for idx, reducer in r._reducer_head_map.items():
            reducer.start_stream(output_height=ctx.output_heights[idx], output_width=ctx.output_widths[idx],
                                 batch_size=ctx.image.shape[B_DIM], channels=r._tile_output_shapes[idx][C_DIM],
                                 device=r.device, dtype=r.dtype, debug_replay=r.debug_reducer_replay)

    def _stitch(self, outputs, tile_outputs, y, x, sides, mask):
        r = self.runtime
        auxiliary = r._reducer_aux_indices()
        for idx, head in enumerate(tile_outputs):
            if idx in auxiliary:
                continue
            _, loc, trimmed = r._build_stitched_tile_output(idx, head, y, x, sides)
            if idx in r._reducer_head_map:
                indices = r._reducer_input_indices.get(idx, (idx,))
                if indices[0] != idx:
                    continue
                payload, _, box = r._build_common_aligned_reducer_payload(
                    head_idx=idx, tile_outputs=tile_outputs, ordered_indices=indices,
                    tile_input_y=y, tile_input_x=x, sides=sides)
                r._accumulate_reducer_forward_tile(idx, payload, box, y, x, sides, mask)
            else:
                stitch_clipped(outputs[idx], trimmed, int(loc.y), int(loc.x))


class BackwardExecutor:
    def __init__(self, runtime, is_backward_streaming_module):
        self.runtime = runtime
        self.is_backward_streaming_module = is_backward_streaming_module

    def execute(self, plan: StreamingPlan, call: BackwardCall):
        r = self.runtime
        if plan is not r.plan:
            raise ValueError("BackwardExecutor received a plan for a different runtime")
        image = call.image.to(r.device, non_blocking=True) if r.copy_to_gpu else call.image
        if call.mask is not None:
            r._active_reducer_mask, r._active_reducer_mask_image = call.mask, image
            r._prepared_reducer_domain_masks = {}
        elif r._active_reducer_mask_image is None:
            r._active_reducer_mask_image = image
        th, tw = r.tile_shape[H_DIM], r.tile_shape[W_DIM]
        valid_h, valid_w = r._compute_valid_output_sizes()
        output_h, output_w = full_output_sizes(image.shape[H_DIM], image.shape[W_DIM], th, tw,
                                               r._tile_output_shapes, r._output_stride_per_output)
        r._current_output_heights, r._current_output_widths = output_h, output_w
        step_h, step_w = r._compute_valid_input_step(valid_h, valid_w)
        rows, cols = r._compute_tile_grid(image.shape[H_DIM], image.shape[W_DIM], th, tw, step_h, step_w)
        grads, spec = r._flatten_output_structure(call.gradient)
        if spec != r._output_spec:
            raise ValueError("Gradient output structure does not match streaming output structure")
        public = r._public_output_indices()
        if len(grads) != len(public):
            raise ValueError(f"Gradient tensor count mismatch: expected={len(public)}, actual={len(grads)}, public_indices={public}")
        internal = [None] * len(r._tile_output_shapes)
        for gradient, idx in zip(grads, public):
            internal[idx] = gradient
        ctx = BackwardExecutionContext(image, internal, th, tw, output_h, output_w)
        tiles = self._tile_iterator(ctx, rows, cols, step_h, step_w)
        r._log_backward_tile_starts(tiles)
        r._validate_backward_tile_iter_matches_forward(tiles)
        r._validate_reducer_lifecycle_for_backward()
        if r.debug_reducer_replay:
            for reducer in r._reducer_head_map.values(): reducer.start_backward_replay()
        last = None
        for y, x, sides in tiles:
            last = sides
            self._replay_tile(ctx, y, x, sides)
        if r.debug_reducer_replay:
            for idx, reducer in r._reducer_head_map.items(): reducer.validate_backward_replay_consumed(head_idx=idx)
        r._saved_tensors = {}
        for module in r.stream_module.modules():
            if self.is_backward_streaming_module(module):
                module.input_loc = None
                module.reset()
        assert last and last.right and last.bottom, "It seems like we could not reconstruct all output"

    def _tile_iterator(self, ctx, rows, cols, step_h, step_w):
        r = self.runtime
        if r._last_forward_tiles:
            return list(r._last_forward_tiles)
        return list(iter_tiles(ctx.image.shape[H_DIM], ctx.image.shape[W_DIM], ctx.tile_height,
                               ctx.tile_width, step_h, step_w, rows, cols))

    def _replay_tile(self, ctx, y, x, sides):
        r = self.runtime
        loc = Box(y, ctx.tile_height, x, ctx.tile_width, sides)
        tile = ctx.image[:, :, y:y + ctx.tile_height, x:x + ctx.tile_width]
        r._saved_tensors = {}
        if not r.copy_to_gpu: tile = tile.to(r.device, non_blocking=True)
        for module in r.stream_module.modules():
            if self.is_backward_streaming_module(module): module.input_loc = loc
        if r.should_normalize: tile = r._normalize_on_gpu(tile)
        if r.gather_input_gradient:
            tile.requires_grad = True
            r.saliency_old_indices = copy.deepcopy(r.saliency_input_module.seen_indices)
        if r.device.type == "cuda" and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=r.dtype): value = r.stream_module(tile)
        else:
            value = r.stream_module(tile)
        tile_outputs, _ = r._flatten_output_structure(value)
        outputs, gradients = [], []
        for idx, head in enumerate(tile_outputs):
            if ctx.gradients[idx] is None: continue
            pair = self._backward_pair(idx, head, tile_outputs, ctx.gradients[idx], y, x, sides, ctx)
            outputs.append(pair[0]); gradients.append(pair[1])
        torch.autograd.backward(outputs, gradients)

    def _backward_pair(self, idx, head, tile_outputs, head_grad, y, x, sides, ctx):
        r = self.runtime
        lost = r._get_tile_lost_for_sides(sides, r._tile_output_lost[idx])
        tile_h, tile_w = r._tile_output_shapes[idx][H_DIM], r._tile_output_shapes[idx][W_DIM]
        stride = r._output_stride_per_output[idx]
        oy, ox = y // int(stride[1]), x // int(stride[2])
        reducer = idx in r._reducer_head_map
        if sides.bottom: oy = max((ctx.output_heights[idx] if reducer else head_grad.shape[H_DIM]) - tile_h, 0)
        if sides.right: ox = max((ctx.output_widths[idx] if reducer else head_grad.shape[W_DIM]) - tile_w, 0)
        trimmed = r._trim_head_output(head, lost).to(r.device, non_blocking=True)
        if reducer:
            gradient = r._build_reducer_gradient(head_grad)
            return r._build_reducer_backward_pair(idx, trimmed, tile_outputs, gradient, y, x, sides,
                                                  oy + lost.top, ox + lost.left)
        gradient = head_grad[:, :, oy:oy + tile_h, ox:ox + tile_w]
        trimmed_grad = gradient[:, :, lost.top:gradient.shape[H_DIM] - lost.bottom,
                                lost.left:gradient.shape[W_DIM] - lost.right]
        if trimmed_grad.shape[-2:] != trimmed.shape[-2:]:
            assert ctx.image.shape[H_DIM] < r.tile_shape[H_DIM] or ctx.image.shape[W_DIM] < r.tile_shape[W_DIM]
            trimmed_grad = trimmed_grad[:, :, :trimmed.shape[H_DIM], :trimmed.shape[W_DIM]]
        return trimmed, trimmed_grad
