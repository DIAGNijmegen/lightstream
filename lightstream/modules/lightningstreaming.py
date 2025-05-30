import torch

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from typing import Any

from lightstream.core.utils import freeze_normalization_layers, unfreeze_streaming_network
from lightstream.modules.streaming import StreamingModule



class LightningStreamingModule(L.LightningModule):
    def __init__(
        self,
        stream_network: StreamingModule,
    ):
        super().__init__()

        self.stream_network = stream_network.stream_network

    @property
    def tile_size(self):
        return self._tile_size

    @tile_size.setter
    def tile_size(self, new_tile_size: int):
        self._tile_size = new_tile_size

    def _prepare_start_for_streaming(self) -> None:
        # Update streaming to put all the inputs/tensors on the right device
        self.stream_network.device = self.device
        self.stream_network.mean = self.stream_network.mean.to(self.device, non_blocking=True)
        self.stream_network.std = self.stream_network.std.to(self.device, non_blocking=True)
        if self.trainer.precision in ["16-mixed", "16", "16-true"]:
            self.stream_network.dtype = torch.float16
        elif self.trainer.precision in ["bf16-mixed", "bf16", "bf16-true"]:
            self.stream_network.dtype = torch.bfloat16
        elif self.trainer.precision in ["32", "32-true"]:
            self.trainer.dtype = torch.float32
        elif self.trainer.precision in ["64", "64-true"]:  # Unlikely to be used, but added for completeness
            self.trainer.dtype = torch.float64
        else:
            self.stream_network.dtype = self.dtype

    def on_train_epoch_start(self) -> None:
        """on_train_epoch_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self.freeze_normalization_layers()

    def on_train_start(self):
        """on_train_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self._prepare_start_for_streaming()

    def on_validation_start(self):
        """on_validation_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self._prepare_start_for_streaming()

    def on_test_start(self):
        """on_test_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self._prepare_start_for_streaming()

    def on_predict_start(self):
        """on_predict_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self._prepare_start_for_streaming()

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.Tensor
        The input tensor in [1,C,H,W] format

        Returns
        -------
        out: torch.Tensor
        The output of the streaming model

        """
        return self.stream_network.forward(x)

    def backward_streaming(self, image, grad):
        self.stream_network.backward(image, grad)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError

    def configure_optimizers(self) -> OptimizerLRScheduler:
        raise NotImplementedError

    def disable_streaming_hooks(self):
        self.stream_network.disable()

    def enable_streaming_hooks(self):
        """Enable streaming hooks and use streamingconv2d modules"""
        self.stream_network.enable()

    def freeze_normalization_layers(self) -> None:
        freeze_normalization_layers(self.stream_network)

    def unfreeze_streaming_network(self):
        unfreeze_streaming_network(self.stream_network)


    def configure_tile_stride(self):
        """
        Helper function that returns the tile stride during streaming.

        Streaming assumes that the input image is perfectly divisible with the network output stride or the
        tile stride. This function will return the tile stride, which can then be used within data processing pipelines
        to pad/crop images to a multiple of the tile stride.

        Examples:

        Returns
        -------
        tile_stride: numpy.ndarray
            the tile stride.


        """
        stride = self.tile_size - (
            self.stream_network.tile_gradient_lost.left + self.stream_network.tile_gradient_lost.right
        )
        stride = stride // self.stream_network.output_stride[-1]
        stride *= self.stream_network.output_stride[-1]
        return stride.detach().cpu().numpy()
