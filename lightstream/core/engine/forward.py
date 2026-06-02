"""Tiled forward execution for the streaming engine."""
import logging

import torch
import torch.backends

from lightstream.core.scnn.utils import B_DIM, C_DIM, H_DIM, W_DIM, Box, Lost
from .config import ForwardContext

logger = logging.getLogger(__name__)

class ForwardMixin:
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

    def _prepare_forward_outputs(self, image, output_heights, output_widths, result_device):
        outputs = [None] * len(self._tile_output_shapes)

        def allocate_non_reducer_outputs():
            aux_indices = self._reducer_aux_indices()
            for idx in range(len(self._tile_output_shapes)):
                if idx in self._reducer_head_map:
                    continue
                if idx in aux_indices:
                    continue
                if outputs[idx] is not None:
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

    def _stitch_forward_outputs(self, outputs, tile_outputs, tile_input_y, tile_input_x, sides, user_mask):
        reducer_aux_indices = self._reducer_aux_indices()
        for head_idx, head_output in enumerate(tile_outputs):
            if head_idx in reducer_aux_indices:
                continue
            _, output_loc, trimmed_output = self._build_stitched_tile_output(
                head_idx=head_idx,
                head_output=head_output,
                tile_input_y=tile_input_y,
                tile_input_x=tile_input_x,
                sides=sides,
            )

            if head_idx in self._reducer_head_map:
                expected_indices = self._reducer_input_indices.get(head_idx, (head_idx,))
                if expected_indices[0] != head_idx:
                    continue
                reducer_payload, _common_loc, common_dst_box = self._build_common_aligned_reducer_payload(
                    head_idx=head_idx,
                    tile_outputs=tile_outputs,
                    ordered_indices=expected_indices,
                    tile_input_y=tile_input_y,
                    tile_input_x=tile_input_x,
                    sides=sides,
                )
                self._accumulate_reducer_forward_tile(
                    head_idx=head_idx,
                    trimmed_payload=reducer_payload,
                    dst_box=common_dst_box,
                    tile_input_y=tile_input_y,
                    tile_input_x=tile_input_x,
                    sides=sides,
                    user_mask=user_mask,
                )
                continue

            self._stitch_non_reducer_output(outputs, head_idx, trimmed_output, output_loc)

    def forward(self, image, result_on_cpu=False, mask=None):
        """Perform forward pass with lightstream."""
        if self.copy_to_gpu:
            image = image.to(self.device, non_blocking=True)
        self._active_reducer_mask = self._normalize_reducer_mask(mask, image)

        plan = self.compiled_plan
        tile_shape = plan.tile_plan.tile_shape if plan.tile_plan is not None else self.tile_shape
        output_spec = plan.public_output_spec if plan.public_output_spec is not None else self._output_spec
        tile_height = tile_shape[H_DIM]
        tile_width = tile_shape[W_DIM]

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
            self.saliency_map = torch.zeros(image.shape, dtype=self.dtype, device="cpu")

        self._last_forward_tiles = []
        logger.debug(
            "Forward tiling step: valid_input_height=%s, valid_input_width=%s, tiles=%sx%s=%s",
            valid_input_height,
            valid_input_width,
            n_rows,
            n_cols,
            n_rows * n_cols,
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
            compiled_plan=plan,
        )
        self._reducer_head_map = {}
        self._reducer_input_indices = {}
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
                    for head_idx, reducer in self._reducer_head_map.items():
                        reducer.start_stream(
                            output_height=forward_ctx.output_heights[head_idx],
                            output_width=forward_ctx.output_widths[head_idx],
                            batch_size=forward_ctx.image.shape[B_DIM],
                            channels=self._tile_output_shapes[head_idx][C_DIM],
                            device=self.device,
                            dtype=self.dtype,
                            debug_replay=self.debug_reducer_replay,
                        )
                    reducers_initialized = True

                if torch.backends.cudnn.benchmark:
                    torch.cuda.empty_cache()

                self._stitch_forward_outputs(
                    outputs,
                    tile_outputs,
                    input_y,
                    input_x,
                    sides,
                    user_mask=self._active_reducer_mask,
                )
                del tile

        assert last_sides is not None and last_sides.bottom and last_sides.right, (
            "It seems like we could not reconstruct all output"
        )

        self._validate_reducer_head_map_resolved()
        self._refresh_compiled_plan()

        del image
        self._saved_tensors = {}
        for idx, reducer in self._reducer_head_map.items():
            outputs[idx] = reducer.finish_stream().to(result_device)

        public_indices = self._public_output_indices()
        self._validate_public_output_indices(public_indices)
        expected_flat_outputs = self._count_tensors_in_spec(output_spec)
        if len(public_indices) != expected_flat_outputs:
            raise RuntimeError(
                f"Public output index count mismatch: expected={expected_flat_outputs}, "
                f"actual={len(public_indices)}; {self._public_output_debug_context(public_indices)}"
            )
        self._validate_public_forward_outputs(outputs, public_indices)
        materialized_outputs = [outputs[idx] for idx in public_indices]

        output, final_idx = self._unflatten_output_structure(materialized_outputs, output_spec)
        assert final_idx == len(materialized_outputs)
        return output

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
