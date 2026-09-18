import torch


class StatisticsProbe(torch.nn.Identity):
    """Expose an otherwise module-free tensor to streaming statistics hooks.

    The probe is an identity during ordinary execution.  StreamingCNN recognizes
    it as a spatially preserving pointwise module while it gathers tile support,
    allowing support on both sides of Python arithmetic to remain visible.
    """

