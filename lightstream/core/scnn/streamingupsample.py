import torch
import torch.nn.functional as F


class StreamingUpsample2d(torch.nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="bilinear", align_corners=None):
        super().__init__()

        if size is not None and scale_factor is not None:
            raise ValueError("StreamingUpsample2d expects either size or scale_factor, not both.")
        if size is None and scale_factor is None:
            raise ValueError("StreamingUpsample2d requires either size or scale_factor.")

        supported_modes = {"bilinear", "nearest"}
        if mode not in supported_modes:
            raise ValueError(f"Unsupported upsample mode '{mode}'. Supported modes: {sorted(supported_modes)}")

        if align_corners is True:
            raise ValueError("StreamingUpsample2d does not support align_corners=True.")

        if mode == "bilinear" and align_corners not in (None, False):
            raise ValueError("StreamingUpsample2d only supports align_corners=None or False for bilinear mode.")

        if mode == "nearest" and align_corners is not None:
            raise ValueError("align_corners is only valid for bilinear mode.")

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, inpt):
        return F.interpolate(
            inpt,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
