"""
https://github.com/Nexuslkl/Swin_MIL/blob/main/models/swin_mil.py

"""
import torch
from torch import Tensor
import torch.nn as nn

from lightstream.models.segment.resnet import make_resnet_backbone
from lightstream.core.reducer import (
    NGWPReducer,
    SizeFocalReducer,
    SigmoidAttentionPoolingReducer,
    AttentionGeMReducer
)
from torchinfo import summary




class WSS(nn.Module):
    "Streaming application of SWIN MIL: https://github.com/Nexuslkl/Swin_MIL"

    def __init__(
        self,
        encoder: str,
        weights: str = "default",
        remove_last_block: bool = True,
        reducer_accumulator_dtype: torch.dtype | None = None,
    ):
        super(WSS, self).__init__()
        self.backbone, self.channels = make_resnet_backbone(
            encoder, weights=weights, include_layer4=not remove_last_block
        )

        self.red1 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype, mask_resize=True)
        self.red2 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype, mask_resize=True)
        self.red3 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype, mask_resize=True)
        self.red4 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype, mask_resize=True)

        self.sigmoid = nn.Sigmoid()
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
        )


        self.w = [0.3, 0.4, 0.3]

    def forward(self, x, mask: torch.Tensor | None = None):
        x1, x2, x3 = self.backbone(x)

        m1 = self.decoder1(x1)
        m2 = self.decoder2(x2)
        m3 = self.decoder3(x3)

        m = 0.3 * m1 + 0.4 * m2 + 0.3 * m3

        return (
            self.red1(self.sigmoid(m1), m1, mask=mask),
            self.red2(self.sigmoid(m2), m2, mask=mask),
            self.red3(self.sigmoid(m3), m3, mask=mask),
            self.red4(self.sigmoid(m), m, mask=mask),
        )


if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 480, 480)).to("cuda")
    network = WSS("resnet18")
    network.to("cuda")

    out_streaming = network(img)
    print(out_streaming)

    summary(network, (1, 3, 480, 480), depth=6)
