"""
This file contains the StreamingConstructor class that converts existing CNN networks into networks capable
of streaming large inputs. During the creation of streaming layers in scnn.py, lost statistics are calculated so that
only the correct parts of the input are considered when calculating gradients. Such an approach is necessary due to
many networks having padding, which will create wrong results when tiles are streamed which should not be padded.

However, only convolutional and local pooling layers need to be used for calculating streaming statistics
since they will have padding. Most other modules (normalization layers, fully connected layers) are not compatible
with streaming or will be kept on module.eval() during both training and inference.

"""

import torch
import torch.nn as nn

from copy import deepcopy
from lightstream.core.scnn import StreamingCNN
from typing import Callable, Optional, Any


class StreamingConstructor:
    def __init__(
        self,
        model: nn.Module,
        tile_size: int,
        verbose: bool = True,
        deterministic: bool = False,
        saliency: bool = False,
        copy_to_gpu: bool = False,
        statistics_on_cpu: bool = True,
        normalize_on_gpu: bool = True,
        mean: Optional[tuple[float, float, float]] = None,
        std: Optional[tuple[float, float, float]] = None,
        tile_cache: Optional[dict] = None,
        add_keep_modules: Optional[list[nn.Module]] = None,
        before_streaming_init_callbacks: Optional[list[Callable[..., Any]]] = None,
        after_streaming_init_callbacks: Optional[list[Callable[..., Any]]] = None,
    ):
        self.model = model
        self.model_copy = deepcopy(self.model)
        self.state_dict = self.save_parameters()

        self.tile_size = tile_size
        self.verbose = verbose
        self.deterministic = deterministic
        self.saliency = saliency
        self.copy_to_gpu = copy_to_gpu
        self.statistics_on_cpu = statistics_on_cpu
        self.normalize_on_gpu = normalize_on_gpu
        self.mean = mean
        self.std = std
        self.tile_cache = tile_cache

        self.before_streaming_init_callbacks = before_streaming_init_callbacks or []
        self.after_streaming_init_callbacks = after_streaming_init_callbacks or []

        self.keep_modules = [
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.AvgPool1d,
            torch.nn.AvgPool2d,
            torch.nn.AvgPool3d,
            torch.nn.MaxPool1d,
            torch.nn.MaxPool2d,
            torch.nn.MaxPool3d,
        ]

        if add_keep_modules is not None:
            self.add_modules_to_keep(add_keep_modules)

        if not self.statistics_on_cpu:
            # Move to cuda manually if statistics are computed on gpu
            device = torch.device("cuda")
            self.model.to(device)

    def add_modules_to_keep(self, module_list: list) -> None:
        """Add extra layers to keep during streaming tile calculations

        Modules in the keep_modules list will not be set to nn.Identity() during streaming initialization
        Parameters
        ----------
        module_list : list
            A list of torch modules to add to the keep_modules list.
        """

        self.keep_modules.extend(module_list)

    def prepare_streaming_model(self) -> nn.Module:
        """Run pre and postprocessing for tile lost calculations
        Returns
        -------
        sCNN : torch.nn.modules
            The streaming module
        """

        # If tile cache is available, it has already been initialized successfully once
        if self.tile_cache:
            return self.create_streaming_model()

        print("")
        # Prepare for streaming tile statistics calculations
        print("Converting modules to nn.Identity()")
        self.convert_to_identity(self.model)
        # execute any callbacks that further preprocess the model
        print("Executing pre-streaming initialization callbacks (if any):")
        self._execute_before_callbacks()

        print("Initializing streaming model")
        sCNN = self.create_streaming_model()

        # check self.stream_network, and reload the proper weights
        print("Restoring model weights")
        self.restore_model_layers(self.model_copy, sCNN.stream_module)
        sCNN.stream_module.load_state_dict(self.state_dict)

        print("Executing post-streaming initialization callbacks (if any):")
        self._execute_after_callbacks()
        return sCNN

    def _execute_before_callbacks(self) -> None:
        for cb_func in self.before_streaming_init_callbacks:
            print(f"Executing callback function {cb_func}")
            cb_func(self.model)
        print("")

    def _execute_after_callbacks(self):
        for cb_func in self.after_streaming_init_callbacks:
            print(f"Executing callback function {cb_func}")
            cb_func(self.model)
        print("")

    def create_streaming_model(self) -> nn.Module:
        return StreamingCNN(
            self.model,
            tile_shape=(1, 3, self.tile_size, self.tile_size),
            deterministic=self.deterministic,
            saliency=self.saliency,
            copy_to_gpu=self.copy_to_gpu,
            verbose=self.verbose,
            statistics_on_cpu=self.statistics_on_cpu,
            normalize_on_gpu=self.normalize_on_gpu,
            mean=self.mean,
            std=self.std,
            state_dict=self.tile_cache,
        )

    def save_parameters(self) -> dict:
        state_dict = self.model.state_dict()
        state_dict = deepcopy(state_dict)
        return state_dict

    def convert_to_identity(self, model: torch.nn.modules) -> None:
        """Convert non-conv and non-local pooling layers to identity

        Parameters
        ----------
        model : torch.nn.Sequential
            The model to substitute
        """

        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                # compound module, go inside it
                self.convert_to_identity(module)
                continue

            # if new module is assigned to a variable, e.g. new = nn.Identity(), then it's considered a duplicate in
            # module.named_children used later. Instead, we use in-place assignment, so each new module is unique
            if not isinstance(module, tuple(self.keep_modules)):
                try:
                    n = int(n)
                    model[n] = torch.nn.Identity()
                except ValueError:
                    setattr(model, str(n), torch.nn.Identity())

    def restore_model_layers(self, model_ref: nn.Module, model_rep: nn.Module) -> None:
        """Restore model layers from Identity to what they were before

        This function requires an exact copy of the model (model_ref) before its layers were set to nn.Identity()
        (model_rep)

        Parameters
        ----------
        model_ref : torch.nn.modules
            The copy of the model before it was set to nn.Identity()
        model_rep : torch.nn.modules
            The stream_module attribute within the streaming model that were set to nn.Identity
        """

        for ref, rep in zip(model_ref.named_children(), model_rep.named_children()):
            n_ref, module_ref = ref
            n_rep, module_rep = rep

            if len(list(module_ref.children())) > 0:
                # compound module, go inside it
                self.restore_model_layers(module_ref, module_rep)
                continue

            if isinstance(module_rep, torch.nn.Identity):
                # simple module
                try:
                    n_ref = int(n_ref)
                    model_rep[n_rep] = model_ref[n_ref]
                except (ValueError, TypeError):
                    try:
                        setattr(model_rep, n_rep, model_ref[int(n_ref)])
                    except (ValueError, TypeError):
                        # Try setting it through block dot operations
                        setattr(model_rep, n_rep, getattr(model_ref, n_ref))


if __name__ == "__main__":
    from lightstream.models.resnet.resnet import resnet18
    import torchvision

    print("Starting constructor checks:")
    """
    print("Test 1: ResNet18")
    network = resnet18()
    stream_network, head = split_resnet(network, num_classes=2)
    constructor = StreamingConstructor(stream_network, 1920, verbose=True)
    constructor.add_keep_modules([torch.nn.BatchNorm2d])
    stream_network = constructor.prepare_streaming_model()
    del network, stream_network, constructor
    print("")
    print("Test succesfully passed")
    
    print("Starting test 2: Convnext-tiny")

    network = convnext_tiny()
    stream_network, head = network.features, torch.nn.Sequential(network.avgpool, network.classifier)
    constructor = StreamingConstructor(
        stream_network, 3520, verbose=True, before_streaming_init_callbacks=[set_layer_scale, toggle_stochastic_depth]
    )
    stream_network = constructor.prepare_streaming_model()
    """
