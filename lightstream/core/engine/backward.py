"""Backward tile replay for the streaming engine."""
import copy
import logging
import math

import torch
import torch.autograd

from lightstream.core.scnn.utils import B_DIM, H_DIM, W_DIM, Box, Sides
from lightstream.core.scnn.streamingconv import StreamingConv2d
from lightstream.core.scnn.streamingupsample import StreamingUpsample2d
from .config import BackwardContext

logger = logging.getLogger(__name__)

class BackwardMixin:
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
        for idx, head_output in enumerate(tile_outputs):
            head_grad = backward_ctx.grad_tensors[idx]
            if head_grad is None:
                continue
            paired_output, paired_grad = self._build_head_backward_pair(
                head_idx=idx,
                head_output=head_output,
                tile_outputs=tile_outputs,
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

    def backward(self, image, grad, mask=None):
        """Perform backward pass with lightstream."""
        if self.copy_to_gpu:
            image = image.to(self.device, non_blocking=True)
        if mask is not None:
            self._active_reducer_mask = self._normalize_reducer_mask(mask, image)

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

        public_indices = self._public_output_indices()
        if len(grad_tensors) != len(public_indices):
            raise ValueError(
                f"Gradient tensor count mismatch: expected={len(public_indices)}, "
                f"actual={len(grad_tensors)}, public_indices={public_indices}"
            )

        internal_grad_tensors = [None] * len(self._tile_output_shapes)
        for public_grad, internal_idx in zip(grad_tensors, public_indices):
            internal_grad_tensors[internal_idx] = public_grad

        if len(self._tile_output_shapes) == 1:
            tile_iter = self._prepare_backward_tile_iter_single_head(image, internal_grad_tensors, tile_height, tile_width)
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
            grad_tensors=internal_grad_tensors,
            tile_height=tile_height,
            tile_width=tile_width,
            output_heights=output_heights,
            output_widths=output_widths,
        )

        if self.debug_reducer_replay:
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
        tile_outputs,
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
                tile_outputs=tile_outputs,
                gradient=gradient,
                tile_input_y=tile_input_y,
                tile_input_x=tile_input_x,
                sides=sides,
                output_y=head_output_y + head_lost.top,
                output_x=head_output_x + head_lost.left,
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
        tile_outputs,
        gradient,
        tile_input_y,
        tile_input_x,
        sides,
        output_y,
        output_x,
    ):
        reducer = self._reducer_head_map[head_idx]
        ordered_indices = self._reducer_input_indices.get(head_idx, (head_idx,))
        if len(ordered_indices) == 1:
            payload = trimmed_output
            common_dst_box = (
                int(output_y),
                int(output_y + trimmed_output.shape[H_DIM]),
                int(output_x),
                int(output_x + trimmed_output.shape[W_DIM]),
            )
        else:
            trimmed_payload, _common_loc, common_dst_box = self._build_common_aligned_reducer_payload(
                head_idx=head_idx,
                tile_outputs=tile_outputs,
                ordered_indices=ordered_indices,
                tile_input_y=tile_input_y,
                tile_input_x=tile_input_x,
                sides=sides,
            )
            payload = tuple(t.to(self.device, non_blocking=True) for t in trimmed_payload)

        dst_y0, dst_y1, dst_x0, dst_x1 = common_dst_box
        ref = payload[0] if isinstance(payload, (tuple, list)) else payload
        valid_mask = self._slice_reducer_mask(
            self._active_reducer_mask,
            dst_y0,
            dst_y1,
            dst_x0,
            dst_x1,
            context=f"backward reducer head {head_idx}",
            expected_shape=(ref.shape[H_DIM], ref.shape[W_DIM]),
        )

        reduced_output, reduced_grad = reducer.build_backward_pair(
            payload,
            gradient,
            input_y=int(tile_input_y),
            input_x=int(tile_input_x),
            sides=sides,
            valid_mask=valid_mask,
        )

        return reduced_output, reduced_grad
