"""
https://github.com/Nexuslkl/Swin_MIL/blob/main/models/swin_mil.py

"""
import torch
import torch.nn as nn

from lightstream.models.segment.resnet import make_resnet_backbone


class WSS(nn.Module):
    def __init__(self):
        super(WSS, self).__init__()
        self.backbone, self.channels = make_resnet_backbone("resnet18")

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )

        self.w = [0.3, 0.4, 0.3]


    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        x1 = self.decoder1(x1)
        x2 = self.decoder2(x2)
        x3 = self.decoder3(x3)

        x = self.w[0] * x1 + self.w[1] * x2 + self.w[2] * x3

        return x1, x2, x3, x

if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 480, 480)).to("cuda")
    network = WSS()
    network.to("cuda")

    out_streaming = network(img)