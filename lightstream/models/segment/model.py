"""
https://github.com/Nexuslkl/Swin_MIL/blob/main/models/swin_mil.py

"""
import torch
from torch import Tensor
import torch.nn as nn

from lightstream.models.segment.resnet import make_resnet_backbone
from lightstream.core.reducer import MeanReducer, AttentionGeMReducer, FusedAttentionGeMReducer
from torchinfo import summary




class WSS(nn.Module):
    def __init__(
        self,
        encoder: str,
        weights: str = "default",
        remove_last_block: bool = True,
        reducer_accumulator_dtype: torch.dtype | None = None,
    ):
        super(WSS, self).__init__()
        self.backbone, self.channels = make_resnet_backbone(encoder, weights=weights, include_layer4=not remove_last_block)
        self.red1 = AttentionGeMReducer(r_init=3.0, accumulator_dtype=reducer_accumulator_dtype)
        self.red2 = AttentionGeMReducer(r_init=3.0, accumulator_dtype=reducer_accumulator_dtype)
        self.red3 = AttentionGeMReducer(r_init=3.0, accumulator_dtype=reducer_accumulator_dtype)
        self.red4 = FusedAttentionGeMReducer(r_init=3.0, accumulator_dtype=reducer_accumulator_dtype)


        self.att1 = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )

        self.att2 = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
        )

        self.att3 = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )

        self.w = [0.3, 0.4, 0.3]


    def forward(self, x, mask: torch.Tensor | None = None):
        x1, x2, x3 = self.backbone(x)

        y1 = self.decoder1(x1)
        y2 = self.decoder2(x2)
        y3 = self.decoder3(x3)
        #y = 0.3 * y1 + 0.4*y2 + 0.3*y3

        att1 = self.att1(x1)
        att2 = self.att2(x2)
        att3 = self.att3(x3)

        return (
            self.red1(y1, att1, mask=mask),
            self.red2(y2, att2, mask=mask),
            self.red3(y3, att3, mask=mask),
            self.red4(y1, y2, y3, att1, att2, att3, mask=mask),
        )

if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 480, 480)).to("cuda")
    network = WSS("resnet18")
    network.to("cuda")

    out_streaming = network(img)
    print(out_streaming)

    summary(network, (1,3, 480, 480), depth=6)
