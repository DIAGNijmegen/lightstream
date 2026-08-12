"""
https://github.com/Nexuslkl/Swin_MIL/blob/main/models/swin_mil.py

"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import List


from lightstream.models.segment.resnet import make_resnet_backbone
from lightstream.core.reducer import NGWPReducer
from torchinfo import summary

from lightstream.core.scnn.streamingmerge import StreamingMerge


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

        hidden_channels = max(1, shallow_channels // 8)
        self.rec_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size),
            nn.Conv2d(deep_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, shallow_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            ),
        )

        self.gamma = 0.0 # LayerScale(shape=1, init_value=0.0)

        self.multiply_merge = StreamingMerge("multiply")
        self.add_merge = StreamingMerge("add")

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
        weighted_features = self.multiply_merge(feature_shallow, weights)
        return self.add_merge(feature_shallow, weighted_features)


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

        self.convs = nn.ModuleList(
            nn.Conv2d(in_channel, n_classes, kernel_size=1)
            for in_channel in encoder_channels
        )

    def forward(
        self, features: List[torch.Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        c2, c3, c4, c5 = features[-4:]

        c2_rect = self.blocks[0](c2, c5)
        c3_rect = self.blocks[1](c3, c5)
        c4_rect = self.blocks[2](c4, c5)

        z2 = self.convs[0](c2_rect)
        z3 = self.convs[1](c3_rect)
        z4 = self.convs[2](c4_rect)
        z5 = self.convs[3](c5)

        return z2, z3, z4, z5

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

        self.encoder, self.channels = make_resnet_backbone( encoder, weights=weights, include_layer4=True)


        self.feature_strides = [4, 8, 16, 32]  # Computed dynamically in weiss, static here for testing

        channels = [y for x,y in self.channels.items()]

        self.decoder = SSHRDecoder(
            encoder_channels=channels,
            encoder_strides=self.feature_strides,
            n_classes=1,
            kernel_size=8,
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

    def _init_reducers(self, reducer_accumulator_dtype: torch.dtype | None):
        self.reducers = nn.ModuleList()

        for i in range(len(self.loss_weights)):
            block = NGWPReducer(eps=1, mask_resize=True, accumulator_dtype=reducer_accumulator_dtype)
            self.reducers.append(block)


    def forward(self, x, mask=None):
        features = self.encoder(x)
        logits = self.decoder(features)

        # Probability maps at native branch resolution
        probs = [self.sigmoid(z) for z in logits]

        # Reduce branches at native resolution
        reduced_outputs = tuple(self.reducers[i](z, p, mask=mask) for i, (z, p) in enumerate(zip(logits, probs)))

        # Only enlarge what is needed for spatial fusion
        probs_up = [self.upsample_blocks[i](p) for i, p in enumerate(probs)]

        p_fused = self.segmentation_head(*probs_up, fuse_weights=self.fuse_weights)
        logit_fused = p_fused.logit(eps=1e-6)

        reduced_outputs += (self.reducers[-1](logit_fused, p_fused, mask=mask),)

        return reduced_outputs



if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 1024, 1024)).to("cuda")
    network = SSHR("resnet18")
    network.to("cuda")

    out_streaming = network(img)
    print(out_streaming)

    summary(network, (1, 3, 1024, 1024), depth=6)
