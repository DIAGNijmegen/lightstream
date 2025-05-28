from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.utils import freeze_normalization_layers, unfreeze_streaming_network


class StreamingModule(L.LightningModule):
    def __init__(
        self,
        stream_network: torch.nn.Module,
        tile_size,
        tile_cache_dir: str | Path = None,
        tile_cache_fname: str | None = None,
        **kwargs,
    ):
        super().__init__()

        # StreamingCNN options
        self._tile_size = tile_size
        self.tile_cache_dir = Path.cwd() if tile_cache_dir is None else Path(tile_cache_dir)
        self.tile_cache_fname = f"tile_cache_{tile_size}" if tile_cache_fname is None else Path(tile_cache_fname)
        tile_cache = self.load_tile_cache_if_needed()  # Load tile cache if present

        # Initialize the streaming network
        self.constructor = StreamingConstructor(
            stream_network,
            self.tile_size,
            tile_cache=tile_cache,
            **kwargs,
        )
        self.copy_to_gpu = self.constructor.copy_to_gpu
        self.stream_network = self.constructor.prepare_streaming_model()

        self.save_tile_cache_if_needed()

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
        freeze_normalization_layers(self.stream_network)

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

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def configure_optimizers(self) -> OptimizerLRScheduler:
        raise NotImplementedError

    def freeze_normalization_layers(self) -> None:
        freeze_normalization_layers(self.stream_network)

    def unfreeze_streaming_network(self):
        unfreeze_streaming_network(self.stream_network)

    def disable_streaming_hooks(self):
        """Disable streaming hooks and replace streamingconv2d  with conv2d modules

        This will still use the StreamingCNN backward and forward functions, but the memory gains from gradient
        checkpointing will be turned off.
        """
        self.stream_network.disable()

    def enable_streaming_hooks(self):
        """Enable streaming hooks and use streamingconv2d modules"""
        self.stream_network.enable()

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

    def save_tile_cache_if_needed(self, overwrite: bool = False):
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
            state_dict = torch.load(
                str(tile_cache_loc),
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
        else:
            print("No tile cache found, calculating it now")
            state_dict = None

        return state_dict
