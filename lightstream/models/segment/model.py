"""
https://github.com/Nexuslkl/Swin_MIL/blob/main/models/swin_mil.py

"""
import torch
from torch import Tensor
import torch.nn as nn

from lightstream.models.segment.resnet import make_resnet_backbone
from lightstream.core.reducer import (
    MeanReducer,
    GeMReducer,
    AttentionGeMReducer,
    FusedAttentionGeMReducer,
)
from torchinfo import summary


class GatedAttention(nn.Module):
    """Convolutional implementation of Gated Attention compatible with streaming."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_classes: int,
        scale_factor: int = 1,
    ):
        super(GatedAttention, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = n_classes

        self.sigmoid_branch = nn.Sequential(
            *[nn.Conv2d(in_channels, hidden_channels, kernel_size=1), nn.Sigmoid()]
        )
        self.tanh_branch = nn.Sequential(
            *[nn.Conv2d(in_channels, hidden_channels, kernel_size=1), nn.Tanh()]
        )

        self.att_logits = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

    def forward(self, x: Tensor) -> Tensor:
        sigmoid_att = self.sigmoid_branch(x)
        tanh_att = self.tanh_branch(x)

        dot_product = sigmoid_att * tanh_att

        att_logits = self.att_logits(dot_product)
        return self.upsample(att_logits)


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

        self.red1 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype, uniform_attention_eps=0.2, mask_resize=True)
        self.red2 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype, uniform_attention_eps=0.2, mask_resize=True)
        self.red3 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype, uniform_attention_eps=0.2, mask_resize=True)
        self.red4 = FusedAttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype, uniform_attention_eps=0.2, mask_resize=True)

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

        self.att_1 = GatedAttention(
            in_channels=64, hidden_channels=32, n_classes=1, scale_factor=4
        )
        self.att_2 = GatedAttention(
            in_channels=128, hidden_channels=64, n_classes=1, scale_factor=8
        )
        self.att_3 = GatedAttention(
            in_channels=256, hidden_channels=128, n_classes=1, scale_factor=16
        )

        self.w = [0.3, 0.4, 0.3]

    def forward(self, x, mask: torch.Tensor | None = None):
        x1, x2, x3 = self.backbone(x)

        y1 = self.decoder1(x1)
        y2 = self.decoder2(x2)
        y3 = self.decoder3(x3)
        y = 0.3 * y1 + 0.4 * y2 + 0.3 * y3

        att1 = self.att_1(x1)
        att2 = self.att_2(x2)
        att3 = self.att_3(x3)

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

    summary(network, (1, 3, 480, 480), depth=6)
