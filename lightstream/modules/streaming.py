from __future__ import annotations

import copy
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

from lightstream.core.constructor import StreamingConstructor
from lightstream.core.reducer import BaseReducer


class StreamingModule(nn.Module):
    """Wrap a PyTorch module for tiled streaming inference and backward passes.

    ``StreamingModule`` constructs a streaming representation of ``stream_network``
    with :class:`lightstream.core.constructor.StreamingConstructor`, optionally
    loading and saving tile statistics from a persistent cache. Cache entries are
    validated with lightweight metadata that describes the model class, tile
    shape, public output structure, internal streaming output count, and reducer
    classes.

    Parameters
    ----------
    stream_network : torch.nn.Module
        Model or model fragment to convert into a streaming network.
    tile_size : int
        Spatial height and width of square NCHW tiles used by the streaming
        constructor. The resulting tile shape is ``(1, 3, tile_size, tile_size)``.
    tile_cache_path : str or pathlib.Path, optional
        File path used to load and save tile cache data. If omitted, a default
        cache file is created in the current working directory.
    defer_prepare : bool, default=False
        Delay metadata probing, cache I/O, and conversion until
        :meth:`prepare_streaming_model` is called. This is useful when a
        framework initializes distributed state after constructing modules.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`lightstream.core.constructor.StreamingConstructor`.

    Attributes
    ----------
    tile_cache_metadata : dict
        Metadata signature for the current model and tile configuration.
    constructor : StreamingConstructor
        Constructor object used to build ``stream_network``.
    stream_network : torch.nn.Module
        Converted streaming network returned by the constructor.
    copy_to_gpu : bool
        Whether the constructor copies intermediate tile results to GPU.
    """

    TILE_CACHE_METADATA_KEY = "tile_cache_metadata"
    TILE_CACHE_METADATA_VERSION = 1

    def __init__(
        self,
        stream_network: torch.nn.Module,
        tile_size,
        tile_cache_path: str | Path = None,
        defer_prepare: bool = False,
        **kwargs,
    ):
        """Initialize the streaming wrapper and prepare tile-cache state.

        Parameters
        ----------
        stream_network : torch.nn.Module
            Model or model fragment to convert into a streaming network.
        tile_size : int
            Spatial height and width of square NCHW tiles.
        tile_cache_path : str or pathlib.Path, optional
            File path used to load and save tile cache data.
        defer_prepare : bool, default=False
            Defer all cache and streaming-network preparation.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`lightstream.core.constructor.StreamingConstructor`.
        """
        super().__init__()
        # StreamingCNN options
        self.tile_size = tile_size
        self.tile_cache_path = Path(tile_cache_path) if tile_cache_path else None
        self.tile_cache_dir = Path.cwd() if tile_cache_path is None else self.tile_cache_path.parent
        self.tile_cache_fname = None if tile_cache_path is None else self.tile_cache_path.stem
        # Keep the original model and constructor arguments intact until the
        # distributed runtime (and, in particular, the process rank) is known.
        self._source_stream_network = stream_network
        self._constructor_kwargs = dict(kwargs)
        self._defer_prepare_requested = defer_prepare
        self._is_prepared = False
        self.tile_cache_metadata = None
        self._tile_cache_was_ignored = False
        if not defer_prepare:
            self.prepare_streaming_model()

    def prepare_streaming_model(self) -> None:
        """Prepare the streaming network once, after distributed initialization.

        In a distributed run only global rank zero is allowed to calculate tile
        statistics.  A status object is broadcast before the barrier so a rank
        zero exception cannot leave its peers blocked indefinitely.
        """
        if self._is_prepared:
            return

        source = self._source_stream_network
        self.tile_cache_metadata = self.build_tile_cache_metadata(source)
        distributed = self._distributed_is_initialized() and self._distributed_world_size() > 1
        # Deferred distributed preparation goes straight to the coordinated
        # path: rank zero rechecks under the lock and every peer performs only
        # the single post-barrier load.
        tile_cache = (
            None
            if distributed and self._defer_prepare_requested
            else self.load_tile_cache_if_needed()
        )

        if not distributed:
            self._prepare_streaming_model(source, tile_cache, **self._constructor_kwargs)
            self.save_tile_cache_if_needed(overwrite=self._tile_cache_was_ignored)
            self._is_prepared = True
            return

        rank = self._distributed_rank()  # get_rank is the global, not local, rank
        status = [None]
        if rank == 0:
            try:
                with self._exclusive_tile_cache_lock():
                    tile_cache = self.load_tile_cache_if_needed()
                    self._prepare_streaming_model(source, tile_cache, **self._constructor_kwargs)
                    if tile_cache is None:
                        self.save_tile_cache_if_needed(overwrite=self._tile_cache_was_ignored)
                status[0] = {"ok": True}
            except Exception as exc:  # communicated to peers before re-raising
                status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        # broadcast_object_list is also a failure-propagation handshake.  The
        # fallback retains compatibility with lightweight mocked distributors.
        try:
            torch.distributed.broadcast_object_list(status, src=0)
        except (RuntimeError, ValueError, AttributeError):
            if rank == 0 and status[0] and not status[0]["ok"]:
                raise RuntimeError(f"Rank 0 failed while preparing tile cache: {status[0]['error']}")

        if status[0] and not status[0]["ok"]:
            raise RuntimeError(f"Rank 0 failed while preparing tile cache: {status[0]['error']}")
        self._distributed_barrier()

        if rank != 0:
            tile_cache = self.load_tile_cache_if_needed()
            if tile_cache is None:
                raise RuntimeError(
                    f"Rank {rank} could not load the tile cache written by global rank 0. "
                    "The cache directory must be on storage shared by every rank. "
                    f"Expected cache path: {self._tile_cache_location()}"
                )
            self._prepare_streaming_model(source, tile_cache, **self._constructor_kwargs)
        self._is_prepared = True

    @staticmethod
    def _distributed_is_initialized() -> bool:
        """Return whether ``torch.distributed`` is available and initialized."""
        distributed = getattr(torch, "distributed", None)
        return (
            distributed is not None
            and distributed.is_available()
            and distributed.is_initialized()
        )

    @staticmethod
    def _distributed_rank() -> int:
        """Return the current distributed rank."""
        return torch.distributed.get_rank()

    @staticmethod
    def _distributed_world_size() -> int:
        """Return the current distributed world size."""
        return torch.distributed.get_world_size()

    @staticmethod
    def _distributed_barrier() -> None:
        """Synchronize all initialized distributed ranks."""
        torch.distributed.barrier()

    def _prepare_streaming_model(
        self,
        stream_network: torch.nn.Module,
        tile_cache: dict | None,
        **kwargs,
    ) -> None:
        """Construct and prepare the streaming network with an optional tile cache."""
        self.constructor = StreamingConstructor(
            stream_network,
            self.tile_size,
            tile_cache=tile_cache,
            **kwargs,
        )
        self.copy_to_gpu = self.constructor.copy_to_gpu
        self.stream_network = self.constructor.prepare_streaming_model()

    def _tile_cache_location(self) -> Path:
        """Return the configured cache path, assigning the default file name if needed."""
        if self.tile_cache_fname is None:
            self.tile_cache_fname = self._default_tile_cache_fname()
        return Path(self.tile_cache_dir) / Path(self.tile_cache_fname)

    def _tile_cache_lock_location(self) -> Path:
        """Return the filesystem lock path associated with the tile cache file."""
        return Path(str(self._tile_cache_location()) + ".lock")

    @contextmanager
    def _exclusive_tile_cache_lock(self):
        """Hold an exclusive filesystem lock for cache generation."""
        lock_path = self._tile_cache_lock_location()
        with open(lock_path, "a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _default_tile_cache_fname(self) -> str:
        """Return the default cache file name for the configured tile size.

        Returns
        -------
        str
            Default tile cache file name without a directory component.
        """
        return "tile_cache_" + "1_3_" + str(self.tile_size) + "_" + str(self.tile_size)

    @classmethod
    def _flatten_output_structure(cls, output):
        """Flatten a structured model output and describe its container layout.

        Parameters
        ----------
        output : torch.Tensor or tuple or list or dict
            Output produced by a model forward pass. Nested combinations of
            tensors, tuples, lists, and dictionaries are supported.

        Returns
        -------
        flat : list[torch.Tensor]
            Tensors found in the output, ordered according to the output
            structure. Dictionary keys are visited in sorted order.
        spec : tuple
            Recursive output-structure specification that can be compared across
            model versions.

        Raises
        ------
        TypeError
            If ``output`` contains an unsupported object type.
        """
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
        """Convert a recursive output specification into stable path strings.

        Parameters
        ----------
        spec : tuple
            Recursive output specification returned by
            :meth:`_flatten_output_structure`.
        path : str, default=""
            Path prefix used internally during recursion.

        Returns
        -------
        list[str]
            Flattened public output specification such as
            ``["/tuple[0]:tensor", "/tuple[1]:tensor"]``.

        Raises
        ------
        TypeError
            If ``spec`` contains an unsupported spec kind.
        """
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
        """Toggle reducer passthrough mode for every reducer in a model.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose reducer modules should be updated.
        enabled : bool
            If ``True``, reducers expose their internal passthrough outputs for
            metadata probing. If ``False``, reducers use their normal public
            output behavior.
        """
        for module in model.modules():
            if isinstance(module, BaseReducer):
                module._streaming_passthrough = enabled

    @staticmethod
    def _restore_training_modes(model: nn.Module, training_modes: dict[nn.Module, bool]) -> None:
        """Restore training/evaluation mode for modules after probing.

        Parameters
        ----------
        model : torch.nn.Module
            Model containing the modules to restore. The parameter is retained
            for call-site readability and future extension.
        training_modes : dict[torch.nn.Module, bool]
            Mapping from module object to the ``training`` value it had before
            metadata probing started.
        """
        for module, was_training in training_modes.items():
            module.train(was_training)

    def build_tile_cache_metadata(self, stream_network: torch.nn.Module) -> dict:
        """Build the metadata signature used to validate tile cache files.

        The signature captures model-level properties that affect whether a
        persisted tile cache can be safely reused. The method performs small
        no-gradient probe forwards to record both the public output structure and
        the internal output count exposed when reducers are in passthrough mode.

        Parameters
        ----------
        stream_network : torch.nn.Module
            Model to inspect before it is converted into a streaming network.

        Returns
        -------
        dict
            Metadata dictionary containing ``version``, ``model_class_name``,
            ``tile_shape``, ``flattened_public_output_spec``,
            ``flattened_internal_output_count``, and ``reducer_class_names``.

        Notes
        -----
        The probe tile is capped at 1024x1024 pixels to avoid allocating very large
        tensors while still exercising output/reducer structure. The persisted
        ``tile_shape`` continues to record the configured tile size.
        """
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
            probe_tile_size = min(int(self.tile_size), 1024)
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
        """Compare loaded cache metadata with the current metadata signature.

        Parameters
        ----------
        cached_metadata : dict or None
            Metadata loaded from a tile cache file. ``None`` represents a legacy
            cache file that predates metadata support.

        Returns
        -------
        list[str]
            Human-readable mismatch descriptions. An empty list means the cache
            metadata matches the current model.
        """
        if cached_metadata is None:
            return ["cache has no metadata"]
        mismatches = []
        for key, expected_value in self.tile_cache_metadata.items():
            cached_value = cached_metadata.get(key)
            if cached_value != expected_value:
                mismatches.append(f"{key}: cached={cached_value!r}, expected={expected_value!r}")
        return mismatches

    def save_tile_cache_if_needed(self, overwrite: bool = False):
        """Save tile cache data and validation metadata when needed.

        Parameters
        ----------
        overwrite : bool, default=False
            If ``True``, replace an existing cache file. If ``False``, leave an
            existing cache file untouched.

        Raises
        ------
        NotADirectoryError
            If the configured cache directory does not exist.

        Notes
        -----
        Tile caches are valid only for the same model layout and tile size. This
        method stores cache metadata alongside the streaming statistics so stale
        caches can be detected on future loads.
        """
        write_path = self._tile_cache_location()

        if Path(self.tile_cache_dir).exists():
            if write_path.exists() and not overwrite:
                print("previous tile cache found and overwrite is false, not saving")

            else:
                print(f"writing streaming cache file to {str(write_path)}")
                tile_cache = self.stream_network.get_tile_cache()
                tile_cache[self.TILE_CACHE_METADATA_KEY] = copy.deepcopy(self.tile_cache_metadata)
                # Never expose a partially-written pickle to another rank.
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{write_path.name}.", dir=str(write_path.parent)
                )
                os.close(fd)
                try:
                    torch.save(tile_cache, temporary_name)
                    os.replace(temporary_name, write_path)
                finally:
                    if os.path.exists(temporary_name):
                        os.unlink(temporary_name)

        else:
            raise NotADirectoryError(f"Did not find {self.tile_cache_dir} or does not exist")

    def load_tile_cache_if_needed(self, use_tile_cache: bool = True):
        """Load a compatible tile cache from disk when available.

        Parameters
        ----------
        use_tile_cache : bool, default=True
            Whether to attempt loading the configured tile cache file.

        Returns
        -------
        dict or None
            Loaded tile cache state dictionary when a compatible cache exists;
            otherwise ``None`` so the constructor recomputes tile statistics.

        Notes
        -----
        Cache files without metadata, or with metadata that differs from the
        current model signature, are treated as stale. Stale caches are ignored,
        a warning is printed, and the cache is marked for overwrite after
        recomputation.
        """

        tile_cache_loc = self._tile_cache_location()

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
        """Run a streaming forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to process with the streaming network.
        mask : torch.Tensor, optional
            Optional spatial mask forwarded to reducer-aware streaming models.

        Returns
        -------
        torch.Tensor or tuple or list or dict
            Output produced by ``self.stream_network``.
        """
        if not self._is_prepared:
            raise RuntimeError("StreamingModule.prepare_streaming_model() must be called before forward")
        return self.stream_network(x, mask=mask)

    def backward_streaming(self, image, grad, mask=None):
        """Run the streaming backward pass for a previously computed gradient.

        Parameters
        ----------
        image : torch.Tensor
            Original full-resolution input image used for the forward pass.
        grad : torch.Tensor or tuple or list or dict
            Gradient with respect to the streaming network output.
        mask : torch.Tensor, optional
            Optional spatial mask forwarded to the streaming backward routine.
        """
        self.stream_network.backward(image, grad, mask=mask)
