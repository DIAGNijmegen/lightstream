import torch
import lightning as L

from lightstream.scnn import StreamingCNN


class StreamingModule(L.LightningModule):
    def __init__(self, stream_network, tile_size, use_streaming=True, *args, **kwargs):
        super().__init__()

        self.tile_size = tile_size
        self.use_streaming = use_streaming
        self.stream_network = StreamingCNN(
            stream_network,
            tile_shape=(1, 3, tile_size, tile_size),
            deterministic=kwargs.get("deterministic", True),
            saliency=kwargs.get("saliency", False),
            gather_gradients=kwargs.get("gather_gradients", False),
            copy_to_gpu=kwargs.get("copy_to_gpu", True),
            verbose=kwargs.get("verbose", True),
        )

        if not self.use_streaming:
            self.disable_streaming()

    def freeze_streaming_normalization_layers(self):
        """Do not use normalization layers within lightstream, only local ops are allowed"""
        freeze_layers = [
            l
            for l in self.stream_network.stream_module.modules()
            if isinstance(l, torch.nn.BatchNorm2d)
        ]

        for mod in freeze_layers:
            mod.eval()

    def disable_streaming(self):
        """Disable streaming hooks and replace streamingconv2d  with conv2d modules"""
        self.stream_network.disable()
        self.use_streaming = False

    def enable_streaming(self):
        """Enable streaming hooks and use streamingconv2d modules"""
        self.stream_network.enable()
        self.use_streaming = True

    def _configure_tile_delta(self):
        """Configure tile delta for variable input shapes"""

        delta = self.tile_size - (
            self.stream_network.tile_gradient_lost.left
            + self.stream_network.tile_gradient_lost.right
        )
        delta = delta // self.stream_network.output_stride[-1]
        delta *= self.stream_network.output_stride[-1]

        # if delta < 3000:
        #     delta = (3000 // delta + 1) * delta
        print("DELTA", delta.item())
        return delta.item()

    def forward_streaming(self, x):
        out = (
            self.stream_network(x)
            if self.use_streaming
            else self.stream_network.stream_module(x)
        )
        return out

    def backward_streaming(self, image, gradient):
        """backward only if streaming is turned on. If not, let pytorch do backward via loss.backward()"""
        if self.use_streaming:
            self.stream_network.backward(image, gradient)
