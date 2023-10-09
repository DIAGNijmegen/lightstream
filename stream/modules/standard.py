from lightning.pytorch.utilities.types import OptimizerLRScheduler

from stream.modules.streaming import StreamingModule
from typing import Any, Tuple
import torch


# TODO: Write control flow when stream is turned off

class StandardModule(StreamingModule):
    def __init__(self, stream_net: torch.nn.modules.container.Sequential,
                 head: torch.nn.modules.container.Sequential,
                 tile_size: int,
                 loss_fn: torch.nn.modules.loss,
                 train_streaming_layers=True,
                 use_streaming=True,
                 *args, **kwargs):

        super().__init__(stream_net, tile_size, use_streaming, *args, **kwargs)

        self.head = head
        self.loss_fn = loss_fn
        self.train_streaming_layers = train_streaming_layers

    def on_train_epoch_start(self) -> None:

        print("on train epoch start hook")
        print("Printing model weights and their param/training attributes")
        print("Setting batchnorm layers to eval")
        self.freeze_streaming_normalization_layers()

        for mod in self.stream_network.stream_module:
            if hasattr(mod, "weight"):
                print(mod, mod.weight)
            else:
                print(mod)

    def forward_head(self, x):
        return self.head(x)

    def forward(self, x):
        fmap = self.forward_streaming(x)
        out = self.forward_head(fmap)
        return out

    def training_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        image, target = batch
        fmap = self.forward_streaming(image)

        # Can only be changed when streaming is enabled, otherwise not a lead variable
        if self.use_streaming:
            fmap.requires_grad = True

        output_head = self.forward_head(fmap)
        loss = self.loss_fn(output_head, target)

        return fmap, output_head, loss

    def backward(self, loss: torch.Tensor, batch: Any, fmap: Any, out: Any, *args: Any, **kwargs: Any):
        """

        Parameters
        ----------
        loss : torch.Tensor, the loss of the model
        batch : The NHWC input image to the model
        out: The (fmap, out) output tuple from the training step. Needed for fmap.grad
        args
        kwargs

        Returns
        -------

        """

        # if use_streaming is False, backward through the stream_network is controlled by loss.backward()
        loss.backward()
        input_image, _ = batch
        if self.train_streaming_layers and self.use_streaming:
            self.backward_streaming(input_image, fmap.grad)

    def get_trainable_params(self):
        if self.train_streaming_layers:
            params = list(self.stream_network) + list(self.head)
        else:
            params = list(self.head)
            for param in self.stream_network: param.requires_grad = False
        return params
