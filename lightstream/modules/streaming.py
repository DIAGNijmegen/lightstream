from typing import Any

import torch
import lightning as L

from pathlib import Path

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from lightstream.modules.constructor import StreamingConstructor


class StreamingModule(L.LightningModule):
    def __init__(self, stream_network, tile_size, train_streaming_layers=True, **kwargs):
        super().__init__()
        self.train_streaming_layers = train_streaming_layers
        # self._stream_module = stream_network

        # StreamingCNN options
        self._tile_size = tile_size
        self.tile_cache_dir = kwargs.pop("tile_cache_dir", Path.cwd())
        self.tile_cache_fname = kwargs.pop("tile_cache_fname", None)

        # Load the tile cache state dict if present
        tile_cache = self.load_tile_cache_if_needed()

        # Initialize the streaming network
        self._constructor_opts = kwargs
        self.constructor = StreamingConstructor(
            stream_network, self.tile_size, tile_cache=tile_cache, **self._constructor_opts
        )
        self.copy_to_gpu = self.constructor.copy_to_gpu
        self.stream_network = self.constructor.prepare_streaming_model()

        self.save_tile_cache_if_needed()
        self.params = self.get_trainable_params()

    @property
    def tile_size(self):
        return self._tile_size

    @tile_size.setter
    def tile_size(self, new_tile_size):
        self._tile_size = new_tile_size

    def freeze_streaming_normalization_layers(self):
        """Do not use normalization layers within lightstream, only local ops are allowed"""
        freeze_layers = [
            l
            for l in self.stream_network.stream_module.modules()
            if isinstance(l, (torch.nn.BatchNorm2d, torch.nn.LayerNorm))
        ]

        for mod in freeze_layers:
            mod.eval()

    def on_train_epoch_start(self) -> None:
        """on_train_epoch_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self.freeze_streaming_normalization_layers()

    def prepare_start_for_streaming(self):

        # Update streaming to put all the inputs/tensors on the right device
        self.stream_network.device = self.device
        self.stream_network.mean = self.stream_network.mean.to(self.device, non_blocking=True)
        self.stream_network.std = self.stream_network.std.to(self.device, non_blocking=True)
        if self.trainer.precision == "16-mixed":
            self.stream_network.dtype = torch.float16
        elif self.trainer.precision == "bf16-mixed":
            self.stream_network.dtype = torch.float16
        else:
            self.stream_network.dtype = self.dtype

    def on_validation_start(self):
        """on_validation_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self.prepare_start_for_streaming()

    def on_train_start(self):
        """on_train_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self.prepare_start_for_streaming()


    def on_test_start(self):
        """on_test_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self.prepare_start_for_streaming()


    def on_predict_start(self):
        """on_predict_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self.prepare_start_for_streaming()


    def disable_streaming_hooks(self):
        """Disable streaming hooks and replace streamingconv2d  with conv2d modules

        This will still use the StreamingCNN backward and forward functions, but the memory gains from gradient
        checkpointing will be turned off.
        """
        self.stream_network.disable()

    def enable_streaming_hooks(self):
        """Enable streaming hooks and use streamingconv2d modules"""
        self.stream_network.enable()

    def forward_streaming(self, x):
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

    def backward_streaming(self, image, gradient):
        """Perform the backward pass using the streaming network

        Backward only if streaming is turned on.
        This method is primarily a convenience function

        Parameters
        ----------
        image: torch.Tensor
            The input image in [1,C,H,W] format
        gradient: torch.Tensor
            The gradient of the next layer in the model to continue backpropagation with

        Returns
        -------

        """

        # If requires_grad is set to false, .backward() in streaming causes errors or overhead, so use a bool
        if self.train_streaming_layers:
            self.stream_network.backward(image, gradient)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

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

    def get_trainable_params(self):
        """Get trainable parameters for the entire model

        If self.streaming_layers is True, then the parameters of the streaming network will be trained.
        Otherwise, the parameters will be left untrained (no gradients will be collected)

        """

        if self.train_streaming_layers:
            params = list(self.stream_network.stream_module.parameters())
            return params
        else:
            print("WARNING: Streaming network will not be trained")
            for param in self.stream_network.stream_module.parameters():
                param.requires_grad = False


    def _remove_streaming_network(self):
        """Converts the streaming network into a non-streaming network

        The former streaming encoder can be addressed as self.stream_network
        This function is currently untested and breaks the class, since there is no way to rebuild the streaming network
        other than calling a new class directly.

        """

        # Convert streamingConv2D into regular Conv2D and turn off streaming hooks
        self.disable_streaming_hooks()
        temp = self.stream_network.stream_module

        # torch modules cannot be overridden normally, so delete and reassign
        del self.stream_network
        self.stream_network = temp

    def save_tile_cache_if_needed(self, overwrite=False):
        """
        Writes the tile cache to a file, so it does not have to be recomputed

        The tile cache is normally calculated for each run.
        However, this can take a long time. By writing it to a file it can be reloaded without the need
        for recomputation.

        Limitations:
        This only works for the exact same model and for a single tile size. If the streaming part of the model
        changes, or if the tile size is changed, it will no longer work.

        """
        if self.tile_cache_fname is None:
            self.tile_cache_fname = "tile_cache_" + "1_3_" + str(self.tile_size) + "_" + str(self.tile_size)
        write_path = Path(self.tile_cache_dir) / Path(self.tile_cache_fname)

        if Path(self.tile_cache_dir).exists():
            if write_path.exists() and not overwrite:
                print("previous tile cache found and overwrite is false, not saving")

            elif self.global_rank == 0:
                print(f"writing streaming cache file to {str(write_path)}")
                torch.save(self.stream_network.get_tile_cache(), str(write_path))

            else:
                print("")
        else:
            raise NotADirectoryError(f"Did not find {self.tile_cache_dir} or does not exist")

    def load_tile_cache_if_needed(self, use_tile_cache: bool = True):
        """
        Load the tile cache for the model from the read_dir

        Parameters
        ----------
        use_tile_cache : bool
            Whether to use the tile cache file and load it into the streaming module

        Returns
        ---------
        state_dict : torch.state_dict | None
            The state dict if present
        """

        if self.tile_cache_fname is None:
            self.tile_cache_fname = "tile_cache_" + "1_3_" + str(self.tile_size) + "_" + str(self.tile_size)

        tile_cache_loc = Path(self.tile_cache_dir) / Path(self.tile_cache_fname)

        if tile_cache_loc.exists() and use_tile_cache:
            print("Loading tile cache from", tile_cache_loc)
            state_dict = torch.load(str(tile_cache_loc), map_location=lambda storage, loc: storage)
        else:
            print("No tile cache found, calculating it now")
            state_dict = None

        return state_dict
