"""GeM reducer API entry points for streaming integration."""

import torch

from .base import StreamingReducer


class StreamingGeMReducer(StreamingReducer):
    """Compatibility streaming reducer entry point for GeM-style reducers.

    Notes
    -----
    This class currently reuses mean-style streaming accumulation behavior.
    """

    def __init__(self, accumulator_dtype: torch.dtype | None = None):
        """Create a GeM reducer placeholder.

        Parameters
        ----------
        accumulator_dtype : torch.dtype | None, default=None
            Optional accumulator dtype used by base streaming logic.
        """
        super().__init__(mode="mean", accumulator_dtype=accumulator_dtype)
