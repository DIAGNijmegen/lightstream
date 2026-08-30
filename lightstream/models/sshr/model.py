"""
https://github.com/Nexuslkl/Swin_MIL/blob/main/models/swin_mil.py

"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from typing import List
from torchinfo import summary

from lightstream.core.reducer import MeanReducer, NormalizedSigmoidAttentionReducer, AttentionKLDivergenceReducer
from lightstream.models.segment.resnet import make_resnet_backbone
from lightstream.core.scnn.streamingmerge import StreamingMerge
from lightstream.core.scnn.streaminglayerscale import LayerScale


class GatedAttention(nn.Module):
    """Convolutional implementation of Gated Attention compatible with streaming."""

    def __init__(
        self,
        embed_channels: int,
        in_channels: int,
        hidden_channels: int,
        n_classes: int,
    ):
        super(GatedAttention, self).__init__()
        self.embed_channels = embed_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = n_classes

        self.bottleneck = nn.Conv2d(
            in_channels=embed_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            bias=False,
        )
        self.sigmoid_branch = nn.Sequential(*[nn.Conv2d(in_channels, hidden_channels, kernel_size=1), nn.Sigmoid()])
        self.tanh_branch = nn.Sequential(*[nn.Conv2d(in_channels, hidden_channels, kernel_size=1), nn.Tanh()])

        self.att_logits = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)
        self.multiply_merge = StreamingMerge("multiply")

    def forward(self, x: Tensor) -> Tensor:
        x = self.bottleneck(x)
        sigmoid_att = self.sigmoid_branch(x)
        tanh_att = self.tanh_branch(x)

        dot_product = self.multiply_merge(sigmoid_att, tanh_att)

        att_logits = self.att_logits(dot_product)
        return att_logits


class LocalRectification(nn.Module):
    """
    Inspired by two articles:
     1. Single-Stage Hierarchical Rectification for Weakly Supervised Histopathology Segmentation
     2. Tiled Squeeze-and-Excite: Channel Attention With Local Spatial Context

    """

    def __init__(
        self,
        shallow_channels: int,
        deep_channels: int,
        scale_factor: int,
        kernel_size: int = 16,
    ):
        super(LocalRectification, self).__init__()

        hidden_channels = 256
        self.rec_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size),
            nn.Conv2d(deep_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, shallow_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
        )

        self.multiply = StreamingMerge("multiply")
        self.add = StreamingMerge("add")
        self.gamma = LayerScale(shape=1, init_value=1.0)

    def forward(self, feature_shallow: Tensor, feature_deep: Tensor) -> Tensor:
        """

        Parameters
        ----------
        feature_shallow : Tensor
        feature_deep : Tensor

        Returns
        -------

        """

        weights = self.rec_block(feature_deep)
        weighted_features = self.multiply(feature_shallow, weights)
        scaled_features = self.gamma(weighted_features)
        return self.add(feature_shallow, scaled_features)


class SSHRDecoder(nn.Module):
    """Decode layer1--layer4 features from a depth-5 encoder.

    The layer1--layer3 features are locally rectified with layer4 context, then
    one segmentation side branch is emitted for each of layer1, layer2, layer3,
    and layer4. The model statically upsamples these four branch outputs for
    streaming-safe fusion and adds the fused output as a fifth loss component.

    Args:
        encoder_channels: Channel counts for layer1, layer2, layer3, and layer4.
        encoder_strides: Input-relative strides for those same four layers.
        n_classes: Number of channels emitted by every side branch.
        kernel_size: Spatial kernel size used by local rectification pooling.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        encoder_strides: List[int],
        n_classes: int = 1,
        kernel_size: int = 16,
    ):
        super().__init__()

        c2_channels, c3_channels, c4_channels, c5_channels = encoder_channels
        c2_stride, c3_stride, c4_stride, deepest_stride = encoder_strides

        self.blocks = nn.ModuleList()
        for shallow_channels, shallow_stride in zip(
            (c2_channels, c3_channels, c4_channels),
            (c2_stride, c3_stride, c4_stride),
        ):
            block = LocalRectification(
                shallow_channels,
                c5_channels,
                kernel_size * deepest_stride // shallow_stride,
                kernel_size=kernel_size,
            )
            self.blocks.append(block)

        self.att_blocks = nn.ModuleList()
        for shallow_channels in (c2_channels, c3_channels, c4_channels, c5_channels):
            block = GatedAttention(
                embed_channels=shallow_channels,
                in_channels=shallow_channels // 2,
                hidden_channels=shallow_channels // 4,
                n_classes=n_classes,
            )
            self.att_blocks.append(block)

        self.ema_blocks = nn.ModuleList()
        for att_block in self.att_blocks:
            block = AveragedModel(att_block, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)
            # Important for DDP / optimizer
            block.requires_grad_(False)
            self.ema_blocks.append(block)

        self.convs = nn.ModuleList(nn.Conv2d(in_channel, n_classes, kernel_size=1) for in_channel in encoder_channels)

    def forward(self, features: List[torch.Tensor]) -> tuple[tuple[Tensor,...], ...]:
        c2, c3, c4, c5 = features[-4:]

        c2_rect = self.blocks[0](c2, c5)
        c3_rect = self.blocks[1](c3, c5)
        c4_rect = self.blocks[2](c4, c5)

        z2 = self.convs[0](c2_rect)
        z3 = self.convs[1](c3_rect)
        z4 = self.convs[2](c4_rect)
        z5 = self.convs[3](c5)
        instance_logits = (z2, z3, z4, z5)

        att_logits = self.forward_attention([c2_rect, c3_rect, c4_rect, c5])
        ema_logits = self.forward_ema([c2_rect, c3_rect, c4_rect, c5])

        return instance_logits, att_logits, ema_logits

    def forward_attention(self, features: List[torch.Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a2 = self.att_blocks[0](features[0])
        a3 = self.att_blocks[1](features[1])
        a4 = self.att_blocks[2](features[2])
        a5 = self.att_blocks[3](features[3])

        return a2, a3, a4, a5

    def forward_ema(self, features: List[torch.Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a2 = self.ema_blocks[0](features[0])
        a3 = self.ema_blocks[1](features[1])
        a4 = self.ema_blocks[2](features[2])
        a5 = self.ema_blocks[3](features[3])

        return a2, a3, a4, a5


class FuseHead(nn.Module):
    """Fuses logit/sigmoid branches from WSSS branches"""

    def __init__(self, apply_sigmoid=True):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, *args, fuse_weights: list[float]):
        if len(fuse_weights) != len(args):
            raise ValueError(f"fuse_weights and args must have same length, found {len(fuse_weights), len(args)}")

        if self.apply_sigmoid:
            args = [x.sigmoid() for x in args]

        return sum(w * x for w, x in zip(fuse_weights, args))


class SSHR(nn.Module):
    "Streaming application of SWIN MIL: https://github.com/Nexuslkl/Swin_MIL"

    def __init__(
        self,
        encoder: str,
        weights: str = "default",
        reducer_accumulator_dtype: torch.dtype | None = None,
    ):
        super(SSHR, self).__init__()
        fuse_weights_list = [0.1, 0.15, 0.25, 0.5]
        loss_weights_list = [1, 1, 1, 1, 1]

        self.register_buffer("fuse_weights", torch.tensor(fuse_weights_list, dtype=torch.float32))
        self.loss_weights = tuple(loss_weights_list)
        self.sigmoid = torch.nn.Sigmoid()

        self.encoder, self.channels = make_resnet_backbone(encoder, weights=weights, include_layer4=True)

        # Computed dynamically in weiss, static here for testing
        self.feature_strides = [4, 8, 16, 32]

        channels = [y for x, y in self.channels.items()]

        self.decoder = SSHRDecoder(
            encoder_channels=channels, encoder_strides=self.feature_strides, n_classes=1, kernel_size=8
        )

        self.segmentation_head = FuseHead(apply_sigmoid=False)
        self.upsample_blocks = self._init_upsample_blocks(self.feature_strides)
        self._init_reducers(reducer_accumulator_dtype)

    def _init_upsample_blocks(self, scale_factors: list[float]):
        blocks = nn.ModuleList()

        for scale_factor in scale_factors:
            block = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
            blocks.append(block)
        return blocks

    def _init_reducers(self, reducer: str = "ngwp"):
        self.reducers = nn.ModuleList()
        self.ema_reducers = nn.ModuleList()

        for i in range(len(self.fuse_weights)):
            block = AttentionKLDivergenceReducer(mask_resize=True, accumulator_dtype=torch.float64)
            self.ema_reducers.append(block)

        for i in range(len(self.loss_weights)):

            block = NormalizedSigmoidAttentionReducer(mask_resize=True, accumulator_dtype=torch.float64)

            if i == len(self.loss_weights) - 1:
                block = MeanReducer(mask_resize=True)

            self.reducers.append(block)

    def forward(self, x, mask=None):
        features = self.encoder(x)
        logits, att_logits, att_logits_ema = self.decoder(features)

        # Probability maps at native branch resolution
        probs = [self.sigmoid(z) for z in logits]

        # Reduce branches at native resolution
        reduced_outputs = tuple(self.reducers[i](z, a, mask=mask) for i, (z, a) in enumerate(zip(logits, att_logits)))
        reduced_outputs_ema = tuple(self.ema_reducers[i](s, t, mask=mask) for i, (s, t) in enumerate(zip(att_logits, att_logits_ema)))

        # Only enlarge what is needed for spatial fusion
        probs_up = [self.upsample_blocks[i](p) for i, p in enumerate(probs)]

        p_fused = self.segmentation_head(*probs_up, fuse_weights=self.fuse_weights)
        #logit_fused = p_fused.logit(eps=1e-6)

        reduced_outputs += (self.reducers[-1](p_fused, mask=mask),)
        reduced_outputs += reduced_outputs_ema
        return reduced_outputs



if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 1024, 1024)).to("cuda")
    network = SSHR("resnet18")
    network.to("cuda")

    out_streaming = network(img)
    print(out_streaming)

    summary(network, (1, 3, 1024, 1024), depth=6)
