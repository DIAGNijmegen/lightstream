"""
https://github.com/Nexuslkl/Swin_MIL/blob/main/models/swin_mil.py

"""
import torch
from torch import Tensor
import torch.nn as nn

from lightstream.models.segment.resnet import make_resnet_backbone
from torchinfo import summary


class GlobalReducer(nn.Module):
    def __init__(self, r: float = 4.0, eps: float = 1e-12):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0 for the generalized mean reducer.")
        self.r = float(r)
        self.eps = float(eps)


    def aggregate(self, logits: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits to be NCHW, got shape {tuple(logits.shape)}")
        probs = torch.sigmoid(logits)
        mean_p_r = probs.pow(self.r).mean(dim=(-2, -1))
        return mean_p_r.clamp_min(self.eps).pow(1.0 / self.r)

    def forward(self, logits: Tensor) -> Tensor:
        return self.aggregate(logits)



class WSS(nn.Module):
    def __init__(self, encoder: str, weights: str="default", remove_last_block: bool =True):
        super(WSS, self).__init__()
        self.backbone, self.channels = make_resnet_backbone(encoder, weights=weights, include_layer4=not remove_last_block)
        self.reducer = GlobalReducer()
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


    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        y1 = self.decoder1(x1)
        y2 = self.decoder2(x2)
        y3 = self.decoder3(x3)

        y = self.w[0] * y1 + self.w[1] * y2 + self.w[2] * y3

        return y1, y2, y3, y

if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 480, 480)).to("cuda")
    network = WSS("resnet18")
    network.to("cuda")

    out_streaming = network(img)
    print(out_streaming)

    summary(network, (1,3, 480, 480), depth=6)