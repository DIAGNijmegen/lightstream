"""Reducer orchestration helpers for the streaming engine."""
import logging

import torch

from lightstream.core.scnn.utils import Box, H_DIM, W_DIM
from lightstream.core.reducer import BaseReducer

logger = logging.getLogger(__name__)

class ReducerMixin:
    def _normalize_reducer_mask(self, mask: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor | None:
        """Normalize reducer masks for rank, device, and dtype.

        Spatial compatibility is intentionally deferred until a concrete
        reducer tile is sliced because reducer heads may operate in a reduced
        feature-space instead of the original input-image space. 3D [N,H,W]
        and 4D [N,C,H,W] masks keep the existing streaming behavior: all batch
        and channel planes are collapsed to one 2D reducer-domain mask with
        ``torch.any(...)``. Per-sample masks would require keeping these axes
        and extending the reducer APIs.
        """
        if mask is None:
            return None
        if mask.ndim == 2:
            return mask.to(device=self.device, dtype=torch.bool)
        if mask.ndim == 3:
            if mask.shape[0] != image.shape[0]:
                raise ValueError(
                    f"3D mask shape {tuple(mask.shape)} must be [N,H,W] with N={image.shape[0]}; "
                    "H/W must align with the reducer/reduced feature spatial domain."
                )
            return torch.any(mask.to(device=self.device, dtype=torch.bool), dim=0)
        if mask.ndim == 4:
            if mask.shape[0] != image.shape[0]:
                raise ValueError(
                    f"4D mask shape {tuple(mask.shape)} must be [N,C,H,W] with N={image.shape[0]}; "
                    "H/W must align with the reducer/reduced feature spatial domain."
                )
            return torch.any(mask.to(device=self.device, dtype=torch.bool), dim=(0, 1))
        raise ValueError(f"mask must be 2D [H,W], 3D [N,H,W], or 4D [N,C,H,W], got shape={tuple(mask.shape)}")

    def _slice_reducer_mask(
        self,
        mask: torch.Tensor | None,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        context: str,
        expected_shape: tuple[int, int],
    ) -> torch.Tensor | None:
        if mask is None:
            return None

        y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
        mask_h, mask_w = int(mask.shape[-2]), int(mask.shape[-1])
        if y0 < 0 or x0 < 0 or y1 > mask_h or x1 > mask_w or y1 < y0 or x1 < x0:
            raise ValueError(
                f"Reducer mask slice {context} ({y0}:{y1}, {x0}:{x1}) is outside mask bounds "
                f"{tuple(mask.shape[-2:])}. The mask must align with the reducer/reduced feature spatial domain, "
                "not necessarily the original input image."
            )

        sliced = mask[y0:y1, x0:x1]
        if tuple(sliced.shape) != tuple(expected_shape):
            raise ValueError(
                f"Reducer mask slice {context} produced shape {tuple(sliced.shape)}, expected {tuple(expected_shape)}. "
                "The mask must align with the reducer/reduced feature spatial domain, not necessarily the original input image."
            )
        return sliced

    def _set_reducer_passthrough(self, enabled: bool):
        for mod in self.stream_module.modules():
            if isinstance(mod, BaseReducer):
                mod._streaming_passthrough = enabled

    def _reducer_aux_indices(self) -> set[int]:
        aux_indices = set()
        for reducer_head, indices in self._reducer_input_indices.items():
            if reducer_head in self._reducer_head_map:
                aux_indices.update(indices[1:])
        return aux_indices

    def _public_output_indices(self) -> list[int]:
        reducer_aux_indices = self._reducer_aux_indices()
        return [
            idx
            for idx in range(len(self._tile_output_shapes))
            if idx not in reducer_aux_indices
        ]

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
                raise RuntimeError(
                    f"Public output head {idx} was not populated during streaming forward; {context}"
                )
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

        output_id_to_index = {}
        for idx, tensor in enumerate(flat_outputs):
            output_id_to_index.setdefault(id(tensor), idx)
        for reducer in self._streaming_reducers:
            reducer_inputs = getattr(reducer, "_passthrough_inputs", None)
            if reducer_inputs is not None:
                if not isinstance(reducer_inputs, (tuple, list)):
                    raise RuntimeError(f"Reducer {type(reducer).__name__} _passthrough_inputs must be tuple/list, got {type(reducer_inputs)}")
                input_indices = []
                for input_pos, inp in enumerate(reducer_inputs):
                    idx = output_id_to_index.get(id(inp))
                    if idx is None:
                        raise RuntimeError(
                            f"Reducer {type(reducer).__name__} input {input_pos} is not present in flattened outputs; cannot resolve reducer head inputs."
                        )
                    input_indices.append(idx)
                output_index = output_id_to_index.get(id(reducer._passthrough_output))
                if output_index is None:
                    raise RuntimeError(f"Reducer {type(reducer).__name__} output is not present in flattened outputs.")
                self._reducer_head_map[output_index] = reducer
                self._reducer_input_indices[output_index] = tuple(input_indices)
                continue

            if getattr(reducer, "_passthrough_output", None) is None:
                continue
            output_index = output_id_to_index.get(id(reducer._passthrough_output))
            if output_index is not None:
                self._reducer_head_map[output_index] = reducer
                self._reducer_input_indices[output_index] = (output_index,)

    def _build_common_aligned_reducer_payload(
        self,
        *,
        head_idx,
        tile_outputs,
        ordered_indices,
        tile_input_y,
        tile_input_x,
        sides,
    ):
        if not ordered_indices or ordered_indices[0] != head_idx:
            raise RuntimeError(f"Reducer head {head_idx} input index order mismatch: indices={ordered_indices}")

        payload_entries = []
        previous_idx = -1
        for reducer_input_idx in ordered_indices:
            if reducer_input_idx <= previous_idx or reducer_input_idx >= len(tile_outputs):
                raise RuntimeError(
                    f"Reducer head {head_idx} input index order mismatch: "
                    f"indices={ordered_indices} over outputs={len(tile_outputs)}"
                )
            previous_idx = reducer_input_idx
            _, input_loc, input_trimmed = self._build_stitched_tile_output(
                head_idx=reducer_input_idx,
                head_output=tile_outputs[reducer_input_idx],
                tile_input_y=tile_input_y,
                tile_input_x=tile_input_x,
                sides=sides,
            )
            if input_trimmed.ndim != 4:
                raise RuntimeError(
                    f"Reducer head {head_idx} tile input {reducer_input_idx} must be NCHW, "
                    f"got {tuple(input_trimmed.shape)}"
                )
            payload_entries.append((reducer_input_idx, input_loc, input_trimmed))

        common_y0 = max(int(loc.y) for _, loc, _ in payload_entries)
        common_x0 = max(int(loc.x) for _, loc, _ in payload_entries)
        common_y1 = min(int(loc.y) + int(tensor.shape[H_DIM]) for _, loc, tensor in payload_entries)
        common_x1 = min(int(loc.x) + int(tensor.shape[W_DIM]) for _, loc, tensor in payload_entries)
        if common_y1 <= common_y0 or common_x1 <= common_x0:
            boxes = [
                (idx, int(loc.y), int(loc.y) + int(tensor.shape[H_DIM]), int(loc.x), int(loc.x) + int(tensor.shape[W_DIM]))
                for idx, loc, tensor in payload_entries
            ]
            raise RuntimeError(f"Reducer head {head_idx} inputs have no common valid intersection: boxes={boxes}")

        cropped_payload = []
        ref_batch = None
        common_h = common_y1 - common_y0
        common_w = common_x1 - common_x0
        for input_pos, (reducer_input_idx, input_loc, input_trimmed) in enumerate(payload_entries):
            if ref_batch is None:
                ref_batch = input_trimmed.shape[B_DIM]
            elif input_trimmed.shape[B_DIM] != ref_batch:
                raise RuntimeError(
                    f"Reducer head {head_idx} tile input batch mismatch at position {input_pos}: "
                    f"expected N={ref_batch}, got shape={tuple(input_trimmed.shape)}"
                )
            src_y0 = common_y0 - int(input_loc.y)
            src_y1 = src_y0 + common_h
            src_x0 = common_x0 - int(input_loc.x)
            src_x1 = src_x0 + common_w
            cropped = input_trimmed[:, :, src_y0:src_y1, src_x0:src_x1]
            if cropped.shape[H_DIM] != common_h or cropped.shape[W_DIM] != common_w:
                raise RuntimeError(
                    f"Reducer head {head_idx} common crop failed for input {reducer_input_idx}: "
                    f"crop={tuple(cropped.shape)} expected spatial=({common_h}, {common_w})"
                )
            cropped_payload.append(cropped)

        common_loc = Box(common_y0, -1, common_x0, -1, sides)
        return cropped_payload, common_loc, (common_y0, common_y1, common_x0, common_x1)

    def _accumulate_reducer_forward_tile(
        self,
        head_idx,
        trimmed_payload,
        dst_box,
        tile_input_y,
        tile_input_x,
        sides,
        user_mask,
    ):
        if not isinstance(trimmed_payload, (tuple, list)) or len(trimmed_payload) == 0:
            raise RuntimeError(f"Reducer head {head_idx} expects non-empty tuple/list payload, got {type(trimmed_payload)}")
        ref = trimmed_payload[0]
        for i, t in enumerate(trimmed_payload):
            if t.ndim != 4:
                raise RuntimeError(f"Reducer head {head_idx} tile input {i} must be NCHW, got {tuple(t.shape)}")
            if t.shape[0] != ref.shape[0] or t.shape[H_DIM] != ref.shape[H_DIM] or t.shape[W_DIM] != ref.shape[W_DIM]:
                raise RuntimeError(
                    f"Reducer head {head_idx} tile input spatial mismatch after common crop: "
                    f"input0={tuple(ref.shape)} input{i}={tuple(t.shape)}; expected same [N,*,H,W]."
                )
        dst_y0, dst_y1, dst_x0, dst_x1 = (int(v) for v in dst_box)
        payload = trimmed_payload[0] if len(trimmed_payload) == 1 else tuple(trimmed_payload)
        tile_mask = self._slice_reducer_mask(
            user_mask,
            dst_y0,
            dst_y1,
            dst_x0,
            dst_x1,
            context=f"forward reducer head {head_idx}",
            expected_shape=(ref.shape[H_DIM], ref.shape[W_DIM]),
        )
        self._reducer_head_map[head_idx].accumulate_stream_tile(
            trimmed_output=payload,
            tile_y=int(tile_input_y),
            tile_x=int(tile_input_x),
            sides=sides,
            dst_box=(dst_y0, dst_y1, dst_x0, dst_x1),
            user_mask=tile_mask,
        )
