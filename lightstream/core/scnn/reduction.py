import torch.nn as nn


class StreamingReductionHint(nn.Module):
    """No-op module to annotate per-output reduction mode for StreamingCNN.

    Streaming logic remains in :class:`StreamingCNN`; this module only carries
    metadata that is parsed during streaming initialization.
    """

    def __init__(self, mode: str = "none", tag: str | None = None):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"none", "sum", "mean"}:
            raise ValueError(f"Unsupported mode='{mode}'. Expected one of: none, sum, mean")
        self.mode = mode
        self.tag = tag

    def forward(self, x):
        return x

