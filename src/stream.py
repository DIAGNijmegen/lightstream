import torch
import lightning as L

from scnn import StreamingCNN
from abc import abstractmethod

from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler


class StreamingModule(L.LightningModule):
    def __init__(self, network, tile_size, train_streaming_layers=False, *args, **kwargs):
        super().__init__()
        self.train_streaming_layers = train_streaming_layers
        self.stream_network, self.head = self.split_model(network)
        self.sCNN = self.convert_to_streaming_network(self.stream_network, tile_size)

    @abstractmethod
    def split_model(self, network: torch.nn.Sequential) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:

        raise NotImplementedError("Model splits should be defined by the user")

    def freeze_normalization_layers(self):
        """Do not use normalization layers within streaming"""
        freeze_layers = [
            l for l in self.network.modules() if isinstance(l, torch.nn.BatchNorm2d)
        ]
        for mod in freeze_layers:
            mod.eval()

    def convert_to_streaming_network(self, network: torch.nn.Sequential, tile_size: int, **kwargs):
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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:

        # Intermediate feature map, needed for backward in streaming
        fmap = self.sCNN(batch)
        out = self.head(fmap)
        return fmap, out

    def backward(self, loss, batch, out, *args: Any, **kwargs: Any):
        """

        Parameters
        ----------
        loss :
        batch : The NHWC input image to the model
        out: The (fmap, out) output tuple from the training step. Needed for fmap.grad
        args
        kwargs

        Returns
        -------

        """

        loss.backward()

        fmap, out =  out
        if self.train_streaming_layers:
            self.sCNN.backward(batch, fmap.grad[None])

    def forward(self, x):
        out = self.sCNN(x)
        print(self.head)
        return self.head(out)


if __name__ == "__main__":
    print("hi")
