"""Tiling and shape planning helpers for the streaming engine."""
import logging
import math

import torch

from lightstream.core.scnn.utils import Sides, _ntuple, H_DIM, W_DIM
from lightstream.core.scnn.streamingconv import StreamingConv2d

logger = logging.getLogger(__name__)

_triple = _ntuple(3)


class PlannerMixin:
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
