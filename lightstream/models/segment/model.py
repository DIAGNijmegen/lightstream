"""
https://github.com/Nexuslkl/Swin_MIL/blob/main/models/swin_mil.py

"""
import torch
from torch import Tensor
import torch.nn as nn

from lightstream.models.segment.resnet import make_resnet_backbone
from lightstream.models.segment.globalreducer import GlobalReducer
from torchinfo import summary




class WSS(nn.Module):
    def __init__(self, encoder: str, weights: str="default", remove_last_block: bool =True):
        super(WSS, self).__init__()
        self.backbone, self.channels = make_resnet_backbone(encoder, weights=weights, include_layer4=not remove_last_block)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Sigmoid()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
            nn.Sigmoid()
        )

        self.w = [0.3, 0.4, 0.3]

        self.reducer1 = GlobalReducer()
        self.reducer2 = GlobalReducer()
        self.reducer3 = GlobalReducer()


    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        y1 = self.decoder1(x1)
        y2 = self.decoder2(x2)
        y3 = self.decoder3(x3)

        y1 = self.reducer1(y1)
        y2 = self.reducer2(y2)
        y3 = self.reducer3(y3)

        return y1, y2, y3

if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 480, 480)).to("cuda")
    network = WSS("resnet18")
    network.to("cuda")

    out_streaming = network(img)
    print(out_streaming)

    summary(network, (1,3, 480, 480), depth=6)
