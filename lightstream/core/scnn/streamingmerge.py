import torch


class StreamingMerge(torch.nn.Module):
    """A module boundary for spatially aligned elementwise merges.

    Only the two operations used by streaming statistics are supported.  The
    forward expressions are deliberately kept literal so ordinary eager
    execution has precisely PyTorch's add/multiply semantics.
    """

    MODES = ("add", "multiply")

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self._streaming_statistics_mode = False

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-2:] != b.shape[-2:]:
            raise ValueError(
                "StreamingMerge inputs must have compatible spatial shapes; "
                f"got {tuple(a.shape[-2:])} and {tuple(b.shape[-2:])}"
            )
        if self.mode == "add":
            return a + b
        if self._streaming_statistics_mode:
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("StreamingMerge statistics require finite operands")
            a_scale = a.detach().abs().amax().clamp_min(1.0)
            b_scale = b.detach().abs().amax().clamp_min(1.0)
            return (a / a_scale) * (b / b_scale)
        return a * b
