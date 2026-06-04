"""
https://github.com/Nexuslkl/Swin_MIL/blob/main/models/swin_mil.py

"""
from collections.abc import Mapping

import torch
from torch import Tensor
import torch.nn as nn

from lightstream.models.segment.resnet import make_resnet_backbone
from lightstream.core.reducer import MeanReducer, GeMReducer, AttentionGeMReducer, FusedAttentionGeMReducer
from torchinfo import summary


class GatedAttention(nn.Module):
    """Convolutional implementation of Gated Attention compatible with streaming."""

    def __init__(self, in_channels: int, hidden_channels: int, n_classes: int, scale_factor: int=1):
        super(GatedAttention, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = n_classes

        self.sigmoid_branch = nn.Sequential(*[nn.Conv2d(in_channels, hidden_channels, kernel_size=1), nn.Sigmoid()])
        self.tanh_branch = nn.Sequential(*[nn.Conv2d(in_channels, hidden_channels, kernel_size=1), nn.Tanh()])

        self.att_logits = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        sigmoid_att = self.sigmoid_branch(x)
        tanh_att = self.tanh_branch(x)

        dot_product = sigmoid_att * tanh_att

        att_logits = self.att_logits(dot_product)
        return self.upsample(att_logits)


class DiagnosticDecoderBranch(nn.Sequential):
    """WSS decoder branch with stable child names for streaming diagnostics.

    The module intentionally keeps the ``0``/``1``/``2`` child names used by the
    previous ``nn.Sequential`` decoders, preserving parameter names such as
    ``decoder1.0.weight`` while allowing :class:`WSS` to run the branch step by
    step and retain gradients for ``x``, the convolution output, the upsample
    output, and the final sigmoid output.
    """

    def __init__(self, in_channels: int, scale_factor: int):
        super().__init__(
            nn.Conv2d(in_channels, 1, 1),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )


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
        self.backbone, self.channels = make_resnet_backbone(encoder, weights=weights, include_layer4=not remove_last_block)
        self.red1 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype)
        self.red2 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype)
        self.red3 = AttentionGeMReducer(accumulator_dtype=reducer_accumulator_dtype)
        self.red4 = FusedAttentionGeMReducer(r_init=4.0, accumulator_dtype=reducer_accumulator_dtype)

        self.decoder1 = DiagnosticDecoderBranch(64, scale_factor=4)
        self.decoder2 = DiagnosticDecoderBranch(128, scale_factor=8)
        self.decoder3 = DiagnosticDecoderBranch(256, scale_factor=16)

        self.att_1 = GatedAttention(in_channels=64, hidden_channels=32, n_classes=1, scale_factor=4)
        self.att_2 = GatedAttention(in_channels=128, hidden_channels=64, n_classes=1, scale_factor=8)
        self.att_3 = GatedAttention(in_channels=256, hidden_channels=128, n_classes=1, scale_factor=16)

        self.w = [0.3, 0.4, 0.3]

        # Test/diagnostic-only switches used to inspect exact autograd
        # boundaries. They are intentionally opt-in so normal training/inference
        # does not retain intermediate gradients.
        self.capture_red4_boundary_grads = False
        self.red4_boundary_tensors: list[dict[str, Tensor]] = []
        self.capture_decoder_branch_grads = False
        self.decoder_branch_tensors: list[dict[str, Tensor]] = []

    def reset_red4_boundary_grad_capture(self) -> None:
        """Clear retained ``red4`` input tensors captured for diagnostics."""
        self.red4_boundary_tensors.clear()

    def reset_decoder_branch_grad_capture(self) -> None:
        """Clear retained decoder branch tensors captured for diagnostics."""
        self.decoder_branch_tensors.clear()

    def _capture_tensors(
        self, enabled: bool, records: list[dict[str, Tensor]], tensors: Mapping[str, Tensor]
    ) -> None:
        if not enabled:
            return

        captured = {}
        for name, tensor in tensors.items():
            if tensor.requires_grad:
                tensor.retain_grad()
            captured[name] = tensor
        records.append(captured)

    def _capture_red4_boundary_tensors(self, **tensors: Tensor) -> None:
        self._capture_tensors(self.capture_red4_boundary_grads, self.red4_boundary_tensors, tensors)

    def _capture_decoder_branch_tensors(self, **tensors: Tensor) -> None:
        self._capture_tensors(self.capture_decoder_branch_grads, self.decoder_branch_tensors, tensors)

    def forward(self, x, mask: torch.Tensor | None = None):
        x1, x2, x3 = self.backbone(x)

        decoder1_conv_out = self.decoder1[0](x1)
        decoder1_upsample_out = self.decoder1[1](decoder1_conv_out)
        y1 = self.decoder1[2](decoder1_upsample_out)

        decoder2_conv_out = self.decoder2[0](x2)
        decoder2_upsample_out = self.decoder2[1](decoder2_conv_out)
        y2 = self.decoder2[2](decoder2_upsample_out)

        decoder3_conv_out = self.decoder3[0](x3)
        decoder3_upsample_out = self.decoder3[1](decoder3_conv_out)
        y3 = self.decoder3[2](decoder3_upsample_out)

        self._capture_decoder_branch_tensors(
            **{
                "decoder1.x": x1,
                "decoder1.conv_out": decoder1_conv_out,
                "decoder1.upsample_out": decoder1_upsample_out,
                "decoder1.y": y1,
                "decoder2.x": x2,
                "decoder2.conv_out": decoder2_conv_out,
                "decoder2.upsample_out": decoder2_upsample_out,
                "decoder2.y": y2,
                "decoder3.x": x3,
                "decoder3.conv_out": decoder3_conv_out,
                "decoder3.upsample_out": decoder3_upsample_out,
                "decoder3.y": y3,
            }
        )
        #y = 0.3 * y1 + 0.4*y2 + 0.3*y3

        att1 = self.att_1(x1)
        att2 = self.att_2(x2)
        att3 = self.att_3(x3)

        self._capture_red4_boundary_tensors(
            y1=y1,
            y2=y2,
            y3=y3,
            att1=att1,
            att2=att2,
            att3=att3,
        )

        return self.red1(y1, att1, mask=mask), self.red2(y2, att2, mask=mask), self.red3(y3, att3, mask=mask), self.red4(y1,y2,y3,att1,att2,att3,mask=mask)


if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 480, 480)).to("cuda")
    network = WSS("resnet18")
    network.to("cuda")

    out_streaming = network(img)
    print(out_streaming)

    summary(network, (1,3, 480, 480), depth=6)
