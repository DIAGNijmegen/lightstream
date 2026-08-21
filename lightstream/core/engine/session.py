"""Per-invocation state shared by a streaming forward/backward pair."""

from dataclasses import dataclass, field
from typing import Any

import torch

from .reducers import ReducerBinding


@dataclass
class StreamSession:
    """Mutable state for exactly one forward and its optional backward replay.

    A session is created by :class:`ForwardExecutor`, remains pending after the
    forward result is returned, and is consumed exactly once by
    :class:`BackwardExecutor`.  Keeping this state here prevents independent
    invocations from silently overwriting state on ``StreamingCNN``.
    """

    image_shape: tuple[int, ...]
    image_dtype: torch.dtype
    active_reducer_mask: torch.Tensor | None = None
    active_reducer_mask_image: torch.Tensor | None = None
    prepared_reducer_domain_masks: dict[Any, torch.Tensor | None] = field(default_factory=dict)
    output_heights: list[int] | None = None
    output_widths: list[int] | None = None
    forward_tiles: list[tuple[Any, ...]] = field(default_factory=list)
    reducer_head_map: dict[int, Any] = field(default_factory=dict)
    reducer_input_indices: dict[int, tuple[int, ...]] = field(default_factory=dict)
    reducer_bindings: dict[int, ReducerBinding] = field(default_factory=dict)
    saved_tensors: dict[Any, Any] = field(default_factory=dict)
    saliency_map: torch.Tensor | None = None
    saliency_old_indices: Any = None
    reducer_replay_started: bool = False
    consumed: bool = False

    @classmethod
    def for_forward(cls, image: torch.Tensor, mask: torch.Tensor | None) -> "StreamSession":
        return cls(tuple(image.shape), image.dtype, mask, image)

    def validate_backward_image(self, image: torch.Tensor) -> None:
        if tuple(image.shape) != self.image_shape or image.dtype != self.image_dtype:
            raise ValueError(
                "Backward image does not match the pending forward session: "
                f"expected shape={self.image_shape}, dtype={self.image_dtype}; "
                f"got shape={tuple(image.shape)}, dtype={image.dtype}."
            )
