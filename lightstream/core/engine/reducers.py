"""Reducer orchestration at the engine/reducer boundary.

Reducers own their algorithms and accumulator state.  This coordinator owns
only traversal-specific bindings, masks, geometry, and lifecycle plumbing.
"""

from dataclasses import dataclass
from typing import Any

import torch

from .geometry import B_DIM, H_DIM, W_DIM, Box


@dataclass(frozen=True)
class StaticReducerBinding:
    """Reducer identity known when a :class:`StreamingPlan` is built."""

    name: str
    reducer_type: str


@dataclass(frozen=True)
class ReducerBinding:
    """Output/input indices resolved for one streaming invocation."""

    head_index: int
    input_indices: tuple[int, ...]
    reducer: Any


class ReducerCoordinator:
    """Small executor-facing API for reducer-specific streaming mechanics."""

    def __init__(self, runtime):
        self.runtime = runtime

    def resolve(self, outputs, session):
        if session.reducer_bindings or not self.runtime._streaming_reducers:
            return
        by_id = {id(value): index for index, value in enumerate(outputs)}
        for reducer in self.runtime._streaming_reducers:
            inputs = getattr(reducer, "_last_inputs", None)
            if inputs is not None:
                if not isinstance(inputs, (tuple, list)):
                    raise RuntimeError(f"Reducer {type(reducer).__name__} _last_inputs must be tuple/list, got {type(inputs)}")
                indices = []
                for position, value in enumerate(inputs):
                    index = by_id.get(id(value))
                    if index is None:
                        raise RuntimeError(f"Reducer {type(reducer).__name__} input {position} is not present in flattened outputs; cannot resolve reducer head inputs.")
                    indices.append(index)
                head = by_id.get(id(reducer._last_output))
                if head is None:
                    raise RuntimeError(f"Reducer {type(reducer).__name__} output is not present in flattened outputs.")
            else:
                head = by_id.get(id(reducer._last_output)) if reducer._last_output is not None else None
                if head is None:
                    continue
                indices = [head]
            binding = ReducerBinding(head, tuple(indices), reducer)
            session.reducer_bindings[head] = binding
            # Kept as read-compatible diagnostic views for existing callers.
            session.reducer_head_map[head] = reducer
            session.reducer_input_indices[head] = binding.input_indices

    def bindings(self, session):
        return session.reducer_bindings.values()

    def auxiliary_indices(self, session):
        return {i for binding in self.bindings(session) for i in binding.input_indices[1:]}

    def validate_forward(self, session):
        resolved = {binding.reducer for binding in self.bindings(session)}
        if len(resolved) != len(self.runtime._streaming_reducers):
            raise RuntimeError(f"Reducer head mapping incomplete after forward tile sampling: resolved={len(resolved)}, expected={len(self.runtime._streaming_reducers)}")

    def validate_backward(self, session):
        if self.runtime._streaming_reducers and not session.reducer_bindings:
            raise RuntimeError("Reducer backward replay requires prior streaming forward pass to resolve reducer heads.")

    def start(self, session, output_heights, output_widths, batch_size):
        r = self.runtime
        for binding in self.bindings(session):
            i, reducer = binding.head_index, binding.reducer
            reducer.start_stream(output_height=output_heights[i], output_width=output_widths[i], batch_size=batch_size,
                                 channels=r._tile_output_shapes[i][1], device=r.device, dtype=r.dtype,
                                 debug_replay=r.debug_reducer_replay)

    def finish(self, session, outputs, device):
        for binding in self.bindings(session):
            outputs[binding.head_index] = binding.reducer.finish_stream().to(device)

    def prepare_mask(self, session, head, reducer):
        mask = session.active_reducer_mask
        if mask is None:
            return None
        image = session.active_reducer_mask_image
        if image is None:
            raise RuntimeError("Reducer mask preparation requires the active forward/backward image context.")
        if mask.ndim == 2: normalized = mask.to(self.runtime.device, torch.bool)
        elif mask.ndim in (3, 4):
            if mask.shape[0] != image.shape[0]:
                raise ValueError(f"{mask.ndim}D mask shape {tuple(mask.shape)} must have N={image.shape[0]}")
            dims = (0,) if mask.ndim == 3 else (0, 1)
            normalized = torch.any(mask.to(self.runtime.device, torch.bool), dim=dims)
        else: raise ValueError(f"mask must be 2D [H,W], 3D [N,H,W], or 4D [N,C,H,W], got shape={tuple(mask.shape)}")
        expected = (int(session.output_heights[head]), int(session.output_widths[head]))
        if tuple(normalized.shape[-2:]) != expected:
            if not getattr(reducer, "mask_resize", False):
                raise ValueError(f"Reducer mask for head_idx={head} has spatial size {tuple(normalized.shape[-2:])}, expected {expected}. Enable mask_resize=True on the reducer to resize the full user mask into the reducer output domain before tile slicing.")
            mode = getattr(reducer, "mask_resize_mode", "nearest")
            if mode != "nearest": raise ValueError(f"Unsupported reducer mask_resize_mode '{mode}' for head_idx={head}. Only 'nearest' is supported for streaming reducer mask resizing.")
            normalized = torch.nn.functional.interpolate(normalized[None, None].float(), size=expected, mode=mode)[0, 0].bool()
        return normalized

    def _mask_slice(self, session, binding, box, shape, context):
        key = (id(binding.reducer), binding.head_index)
        if key not in session.prepared_reducer_domain_masks:
            session.prepared_reducer_domain_masks[key] = self.prepare_mask(session, binding.head_index, binding.reducer)
        mask = session.prepared_reducer_domain_masks[key]
        if mask is None: return None
        y0, y1, x0, x1 = map(int, box)
        if y0 < 0 or x0 < 0 or y1 > mask.shape[-2] or x1 > mask.shape[-1]:
            raise ValueError(f"Reducer mask slice {context} ({y0}:{y1}, {x0}:{x1}) is outside mask bounds {tuple(mask.shape[-2:])}. The mask must align with the reducer/reduced feature spatial domain, not necessarily the original input image.")
        value = mask[y0:y1, x0:x1]
        if tuple(value.shape) != tuple(shape): raise ValueError(f"Reducer mask slice {context} produced shape {tuple(value.shape)}, expected {tuple(shape)}. The mask must align with the reducer/reduced feature spatial domain, not necessarily the original input image.")
        return value

    def aligned_payload(self, head, tile_outputs, indices, y, x, sides):
        r = self.runtime
        if not indices or indices[0] != head: raise RuntimeError(f"Reducer head {head} input index order mismatch: indices={indices}")
        entries = []
        previous = -1
        for index in indices:
            if index <= previous or index >= len(tile_outputs): raise RuntimeError(f"Reducer head {head} input index order mismatch: indices={indices} over outputs={len(tile_outputs)}")
            previous = index
            _, loc, value = r._build_stitched_tile_output(index, tile_outputs[index], y, x, sides)
            if value.ndim != 4: raise RuntimeError(f"Reducer head {head} tile input {index} must be NCHW, got {tuple(value.shape)}")
            entries.append((index, loc, value))
        y0=max(int(v.y) for _,v,_ in entries); x0=max(int(v.x) for _,v,_ in entries)
        y1=min(int(v.y)+t.shape[H_DIM] for _,v,t in entries); x1=min(int(v.x)+t.shape[W_DIM] for _,v,t in entries)
        if y1<=y0 or x1<=x0: raise RuntimeError(f"Reducer head {head} inputs have no common valid intersection")
        payload=[]
        for index,loc,value in entries:
            sy, sx = y0-int(loc.y), x0-int(loc.x)
            payload.append(value[:,:,sy:sy+y1-y0,sx:sx+x1-x0])
        return payload, Box(y0,-1,x0,-1,sides), (y0,y1,x0,x1)

    def accumulate(self, session, head, tile_outputs, y, x, sides):
        binding=session.reducer_bindings[head]
        payload, _, box=self.aligned_payload(head,tile_outputs,binding.input_indices,y,x,sides)
        ref=payload[0]; mask=self._mask_slice(session,binding,box,ref.shape[-2:],f"forward reducer head {head}")
        binding.reducer.accumulate_stream_tile(trimmed_output=payload[0] if len(payload)==1 else tuple(payload), tile_y=int(y), tile_x=int(x), sides=sides, dst_box=box, user_mask=mask)

    def backward_pair(self, session, head, trimmed, tile_outputs, gradient, y, x, sides, oy, ox):
        binding=session.reducer_bindings[head]
        if len(binding.input_indices)==1: payload=trimmed; box=(oy,oy+trimmed.shape[H_DIM],ox,ox+trimmed.shape[W_DIM])
        else:
            values,_,box=self.aligned_payload(head,tile_outputs,binding.input_indices,y,x,sides)
            payload=tuple(v.to(self.runtime.device,non_blocking=True) for v in values)
        ref=payload[0] if isinstance(payload,(tuple,list)) else payload
        mask=self._mask_slice(session,binding,box,ref.shape[-2:],f"backward reducer head {head}")
        return binding.reducer.build_backward_pair(payload,gradient,input_y=int(y),input_x=int(x),sides=sides,valid_mask=mask)
