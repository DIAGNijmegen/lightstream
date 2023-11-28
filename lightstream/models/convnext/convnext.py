import torch
import torch.nn as nn

from copy import deepcopy
from modules.base import BaseModel
from torchvision.models import convnext_tiny, convnext_small
from torchvision.ops.misc import Permute
import torchvision


# TODO: Integrate this into a basic streaming constructor class
def _toggle_stochastic_depth(model, training=False):
    for m in model.modules():
        if isinstance(m, torchvision.ops.StochasticDepth):
            m.training = training

def _convert_to_identity(model, old):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            _convert_to_identity(module, old)

        # if new module is assigned to a variable, e.g. new = nn.Identity(), then it's considered a duplicate in
        # module.named_children used later. Instead, we use in-place assignment, so each new module is unique
        if isinstance(module, old):
            ## simple module
            try:
                n = int(n)
                model[n] = torch.nn.Identity()
            except:
                setattr(model, str(n), torch.nn.Identity())

def _set_layer_scale(model, val):
    for x in model.modules():
        if hasattr(x, "layer_scale"):
            x.layer_scale.data.fill_(val)

def _restore_layers(model_ref, model_rep):
    for ref, rep in zip(
        model_ref.named_children(), model_rep.named_children()
    ):

        n_ref, module_ref = ref
        n_rep, module_rep = rep

        if len(list(module_ref.children())) > 0:
            ## compound module, go inside it
            _restore_layers(module_ref, module_rep)

        if isinstance(module_rep, torch.nn.Identity):
            ## simple module
            try:
                n_ref = int(n_ref)
                model_rep[n_rep] = model_ref[n_ref]
            except:
                setattr(model_rep, n_rep, model_ref[int(n_ref)])

def _save_parameters(model):
    state_dict = model.state_dict()
    state_dict = deepcopy(state_dict)
    return state_dict


def _prepare_for_streaming_statistics(stream_network):
    # Temporarily replace layernorm, linear, and gelu to not meddle with statistics calculations in streaming
    _convert_to_identity(stream_network, torch.nn.LayerNorm)
    _convert_to_identity(stream_network, torch.nn.Linear)
    _convert_to_identity(stream_network, torch.nn.GELU)
    _convert_to_identity(stream_network, Permute)


    # Do not use stochastic depth when initializing streaming
    _toggle_stochastic_depth(stream_network)

    # Set layer scale parameters to one to not meddle with streaming statistics calculations
    _set_layer_scale(stream_network, 1.0)


class StreamingConvnext(BaseModel):
    model_choices = {"convnext_tiny": convnext_tiny, "convnext_small": convnext_small}

    def __init__(
        self,
        model_name: str,
        tile_size: int,
        loss_fn: torch.nn.functional,
        train_streaming_layers: bool = True,
        use_streaming: bool = True,
        use_stochastic_depth: bool = False,
        *args,
        **kwargs,
    ):
        assert model_name in list(StreamingConvnext.model_choices.keys())

        self.model_name = model_name
        self.use_stochastic_depth = use_stochastic_depth

        network = StreamingConvnext.model_choices[model_name](weights="IMAGENET1K_V1")
        stream_network, head = network.features, torch.nn.Sequential(network.avgpool, network.classifier)

        # Save parameters for easy recovery of module parameters later
        state_dict = _save_parameters(stream_network)

        # Prepare for streaming tile statistics calculations
        _prepare_for_streaming_statistics(stream_network)

        super().__init__(
            stream_network,
            head,
            tile_size,
            loss_fn,
            train_streaming_layers=train_streaming_layers,
            use_streaming=use_streaming,
            *args,
            **kwargs,
        )

        # check self.stream_network, and reload the proper weights
        self._restore_model_layers()

        # re apply layer scale weights and stochastic depth settings
        self.stream_network.stream_module.load_state_dict(state_dict)

        if use_stochastic_depth:
            _toggle_stochastic_depth(stream_network, training=True)


    def _restore_model_layers(self):
        temp_model = StreamingConvnext.model_choices[self.model_name](weights="IMAGENET1K_V1").features
        _restore_layers(temp_model, self.stream_network.stream_module)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    model = StreamingConvnext("convnext_tiny", 1600, nn.MSELoss)
    print(model)
