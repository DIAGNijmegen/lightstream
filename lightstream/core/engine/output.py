"""Output-structure helpers for streaming engine internals."""

from __future__ import annotations

from typing import Any

import torch


def flatten_output_structure(output: Any):
    if isinstance(output, torch.Tensor):
        return [output], ("tensor", None)
    if isinstance(output, tuple):
        flat = []
        children = []
        for x in output:
            child_flat, child_spec = flatten_output_structure(x)
            flat.extend(child_flat)
            children.append(child_spec)
        return flat, ("tuple", children)
    if isinstance(output, list):
        flat = []
        children = []
        for x in output:
            child_flat, child_spec = flatten_output_structure(x)
            flat.extend(child_flat)
            children.append(child_spec)
        return flat, ("list", children)
    if isinstance(output, dict):
        flat = []
        children = []
        for key in sorted(output.keys()):
            child_flat, child_spec = flatten_output_structure(output[key])
            flat.extend(child_flat)
            children.append((key, child_spec))
        return flat, ("dict", children)
    raise TypeError(f"Unsupported output type for streaming: {type(output)}")


def unflatten_output_structure(flat, spec, index=0):
    kind, payload = spec
    if kind == "tensor":
        return flat[index], index + 1
    if kind in {"tuple", "list"}:
        values = []
        for child in payload:
            value, index = unflatten_output_structure(flat, child, index)
            values.append(value)
        return (tuple(values) if kind == "tuple" else values), index
    if kind == "dict":
        values = {}
        for key, child in payload:
            value, index = unflatten_output_structure(flat, child, index)
            values[key] = value
        return values, index
    raise TypeError(f"Unsupported output spec kind: {kind}")


def count_tensors_in_spec(spec) -> int:
    kind, payload = spec
    if kind == "tensor":
        return 1
    if kind in {"tuple", "list"}:
        return sum(count_tensors_in_spec(child) for child in payload)
    if kind == "dict":
        return sum(count_tensors_in_spec(child) for _, child in payload)
    raise TypeError(f"Unsupported output spec kind: {kind}")


def reducer_aux_indices(reducer_input_indices: dict, reducer_head_map: dict) -> set[int]:
    aux_indices = set()
    for reducer_head, indices in reducer_input_indices.items():
        if reducer_head in reducer_head_map:
            aux_indices.update(indices[1:])
    return aux_indices


def public_output_indices(tile_output_shapes, reducer_input_indices: dict, reducer_head_map: dict) -> list[int]:
    aux_indices = reducer_aux_indices(reducer_input_indices, reducer_head_map)
    return [idx for idx in range(len(tile_output_shapes)) if idx not in aux_indices]


def public_output_debug_context(public_indices, reducer_input_indices: dict, reducer_head_map: dict, aux_indices=None) -> str:
    if aux_indices is None:
        aux_indices = reducer_aux_indices(reducer_input_indices, reducer_head_map)
    return (
        f"public_indices={list(public_indices)}, "
        f"reducer_auxiliary_indices={sorted(aux_indices)}, "
        f"self._reducer_input_indices={reducer_input_indices}"
    )


def validate_public_output_indices(public_indices, reducer_input_indices: dict, reducer_head_map: dict) -> None:
    aux_indices = reducer_aux_indices(reducer_input_indices, reducer_head_map)
    leaked_aux_indices = sorted(set(public_indices) & aux_indices)
    if leaked_aux_indices:
        raise RuntimeError(
            "Public output indices include reducer auxiliary indices; "
            f"leaked_auxiliary_indices={leaked_aux_indices}; "
            f"{public_output_debug_context(public_indices, reducer_input_indices, reducer_head_map, aux_indices)}"
        )


def validate_public_forward_outputs(
    outputs,
    public_indices,
    reducer_input_indices: dict,
    reducer_head_map: dict,
    *,
    debug_forward_sentinel_check: bool = False,
) -> None:
    context = public_output_debug_context(public_indices, reducer_input_indices, reducer_head_map)
    for idx in public_indices:
        output = outputs[idx]
        if output is None:
            raise RuntimeError(f"Public output head {idx} was not populated during streaming forward; {context}")
        if debug_forward_sentinel_check and torch.all(output == 999):
            raise RuntimeError(
                f"Public output head {idx} still contains only the unstitched sentinel value 999; {context}"
            )
