from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.reducer import BaseReducer


class StreamingModule(nn.Module):
    TILE_CACHE_METADATA_KEY = "tile_cache_metadata"
    TILE_CACHE_METADATA_VERSION = 1

    def __init__(self, stream_network: torch.nn.Module, tile_size, tile_cache_path: str | Path = None, **kwargs):
        super().__init__()
        # StreamingCNN options
        self.tile_size = tile_size
        self.tile_cache_path = Path(tile_cache_path) if tile_cache_path else None
        self.tile_cache_dir = Path.cwd() if tile_cache_path is None else self.tile_cache_path.parent
        self.tile_cache_fname = None if tile_cache_path is None else self.tile_cache_path.stem
        self.tile_cache_metadata = self.build_tile_cache_metadata(stream_network)
        self._tile_cache_was_ignored = False
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
        self.save_tile_cache_if_needed(overwrite=self._tile_cache_was_ignored)

    def _default_tile_cache_fname(self) -> str:
        return "tile_cache_" + "1_3_" + str(self.tile_size) + "_" + str(self.tile_size)

    @classmethod
    def _flatten_output_structure(cls, output):
        if isinstance(output, torch.Tensor):
            return [output], ("tensor", None)
        if isinstance(output, tuple):
            flat = []
            children = []
            for item in output:
                child_flat, child_spec = cls._flatten_output_structure(item)
                flat.extend(child_flat)
                children.append(child_spec)
            return flat, ("tuple", children)
        if isinstance(output, list):
            flat = []
            children = []
            for item in output:
                child_flat, child_spec = cls._flatten_output_structure(item)
                flat.extend(child_flat)
                children.append(child_spec)
            return flat, ("list", children)
        if isinstance(output, dict):
            flat = []
            children = []
            for key in sorted(output.keys()):
                child_flat, child_spec = cls._flatten_output_structure(output[key])
                flat.extend(child_flat)
                children.append((key, child_spec))
            return flat, ("dict", children)
        raise TypeError(f"Unsupported output type for streaming cache metadata: {type(output)}")

    @classmethod
    def _flatten_output_spec(cls, spec, path: str = "") -> list[str]:
        kind, payload = spec
        if kind == "tensor":
            return [f"{path}:tensor"]
        if kind in {"tuple", "list"}:
            flat_spec = []
            for index, child in enumerate(payload):
                child_path = f"{path}/{kind}[{index}]"
                flat_spec.extend(cls._flatten_output_spec(child, child_path))
            return flat_spec
        if kind == "dict":
            flat_spec = []
            for key, child in payload:
                child_path = f"{path}/dict[{key!r}]"
                flat_spec.extend(cls._flatten_output_spec(child, child_path))
            return flat_spec
        raise TypeError(f"Unsupported output spec kind for streaming cache metadata: {kind}")

    @staticmethod
    def _set_reducer_passthrough(model: nn.Module, enabled: bool) -> None:
        for module in model.modules():
            if isinstance(module, BaseReducer):
                module._streaming_passthrough = enabled

    @staticmethod
    def _restore_training_modes(model: nn.Module, training_modes: dict[nn.Module, bool]) -> None:
        for module, was_training in training_modes.items():
            module.train(was_training)

    def build_tile_cache_metadata(self, stream_network: torch.nn.Module) -> dict:
        """Build a compact model signature used to validate persisted tile caches."""
        reducer_class_names = [
            module.__class__.__name__ for module in stream_network.modules() if isinstance(module, BaseReducer)
        ]
        metadata = {
            "version": self.TILE_CACHE_METADATA_VERSION,
            "model_class_name": stream_network.__class__.__name__,
            "tile_shape": (1, 3, self.tile_size, self.tile_size),
            "flattened_public_output_spec": None,
            "flattened_internal_output_count": None,
            "reducer_class_names": reducer_class_names,
        }

        try:
            first_parameter = next(stream_network.parameters())
            device = first_parameter.device
            dtype = first_parameter.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        training_modes = {module: module.training for module in stream_network.modules()}
        try:
            stream_network.eval()
            probe_tile_size = min(int(self.tile_size), 64)
            tile = torch.ones((1, 3, probe_tile_size, probe_tile_size), dtype=dtype, device=device)
            with torch.no_grad():
                self._set_reducer_passthrough(stream_network, False)
                public_output = stream_network(tile)
                _, public_output_spec = self._flatten_output_structure(public_output)

                self._set_reducer_passthrough(stream_network, True)
                internal_output = stream_network(tile)
                internal_outputs, _ = self._flatten_output_structure(internal_output)

            metadata["flattened_public_output_spec"] = self._flatten_output_spec(public_output_spec)
            metadata["flattened_internal_output_count"] = len(internal_outputs)
        finally:
            self._set_reducer_passthrough(stream_network, False)
            self._restore_training_modes(stream_network, training_modes)

        return metadata

    def _metadata_mismatches(self, cached_metadata: dict | None) -> list[str]:
        if cached_metadata is None:
            return ["cache has no metadata"]
        mismatches = []
        for key, expected_value in self.tile_cache_metadata.items():
            cached_value = cached_metadata.get(key)
            if cached_value != expected_value:
                mismatches.append(f"{key}: cached={cached_value!r}, expected={expected_value!r}")
        return mismatches

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
            self.tile_cache_fname = self._default_tile_cache_fname()
        write_path = Path(self.tile_cache_dir) / Path(self.tile_cache_fname)

        if Path(self.tile_cache_dir).exists():
            if write_path.exists() and not overwrite:
                print("previous tile cache found and overwrite is false, not saving")

            else:
                print(f"writing streaming cache file to {str(write_path)}")
                tile_cache = self.stream_network.get_tile_cache()
                tile_cache[self.TILE_CACHE_METADATA_KEY] = copy.deepcopy(self.tile_cache_metadata)
                torch.save(tile_cache, str(write_path))

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
            self.tile_cache_fname = self._default_tile_cache_fname()

        tile_cache_loc = Path(self.tile_cache_dir) / Path(self.tile_cache_fname)

        if tile_cache_loc.exists() and use_tile_cache:
            print("Loading tile cache from", tile_cache_loc)
            state_dict = torch.load(
                str(tile_cache_loc),
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
            mismatches = self._metadata_mismatches(state_dict.get(self.TILE_CACHE_METADATA_KEY))
            if mismatches:
                print(
                    "Warning: ignoring stale tile cache metadata for "
                    f"{tile_cache_loc}: " + "; ".join(mismatches)
                )
                self._tile_cache_was_ignored = True
                state_dict = None
        else:
            print("No tile cache found, calculating it now")
            state_dict = None

        return state_dict

    def forward(self, x, mask=None):
        return self.stream_network(x, mask=mask)

    def backward_streaming(self, image, grad, mask=None):
        self.stream_network.backward(image, grad, mask=mask)
