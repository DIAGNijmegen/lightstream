import torch
import torch.nn as nn
from pathlib import Path

from lightstream.core.constructor import StreamingConstructor


class StreamingModule(nn.Module):
    def __init__(self, stream_network: torch.nn.Module, tile_size, tile_cache_path: str | Path = None, **kwargs):
        super().__init__()
        # StreamingCNN options
        self.tile_size = tile_size
        self.tile_cache_path = Path(tile_cache_path) if tile_cache_path else None
        self.tile_cache_dir = Path.cwd() if tile_cache_path is None else self.tile_cache_path.parent
        self.tile_cache_fname = None if tile_cache_path is None else self.tile_cache_path.stem
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

            else:
                print(f"writing streaming cache file to {str(write_path)}")
                torch.save(self.stream_network.get_tile_cache(), str(write_path))

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

    def forward(self, x):
        return self.stream_network(x)

    def backward_streaming(self, image, grad):
        self.stream_network.backward(image, grad)