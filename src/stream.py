from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT

from scnn import StreamingCNN
import torch
import lightning as L


class StreamingModule(L.LightningModule):
    def __init__(self, network, tile_size, train_streaming_layers=False, *args, **kwargs):
        super().__init__()
        self.stream_network = self.convert_to_streaming_network(network, tile_size)
        self.train_streaming_layers = train_streaming_layers

    def freeze_normalization_layers(self):
        """Do not use normalization layers within streaming"""
        freeze_layers = [
            l for l in self.network.modules() if isinstance(l, torch.nn.BatchNorm2d)
        ]
        for mod in freeze_layers:
            mod.eval()

    def convert_to_streaming_network(self, network, tile_size, **kwargs):
        stream_net = StreamingCNN(
            network,
            tile_shape=(1, 3, tile_size, tile_size),
            deterministic=kwargs.get("deterministic", True),
            saliency=kwargs.get("saliency", False),
            gather_gradients=kwargs.get("gather_gradients", False),
            copy_to_gpu=kwargs.get("copy_to_gpu", True),
            verbose=kwargs.get("verbose", True),
        )
        return stream_net

    def backward_image(self, x, gradient):
        if self.train_streaming_layers:
            self.stream_network.backward(x, gradient[None])

    def forward(self, x):
        return self.stream_network(x)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass





if __name__ == "__main__":
    print("hi")
