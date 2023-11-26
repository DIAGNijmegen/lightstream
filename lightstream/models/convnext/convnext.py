import torch
import torch.nn as nn

from copy import deepcopy
from modules.base import BaseModel
from torchvision.models import convnext_tiny, convnext_small

import torchvision


def freeze_stochastic_depth(model):
    for m in model.modules():
        if isinstance(m, torchvision.ops.StochasticDepth):
            m.training = False
    return model

def convert_to_identity(model, old):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            convert_to_identity(module, old)

        # if new module is assigned to a variable, e.g. new = nn.Identity(), then it will be considered a duplicate
        # in module.named_children, which is used later. Instead, we use in-place assignment, so each new module
        # is unique
        if isinstance(module, old):
            ## simple module
            try:
                n = int(n)
                model[n] = torch.nn.Identity()
            except:
                setattr(model, str(n), torch.nn.Identity())

def set_layer_scale(model, val):
    for x in model.modules():
        if hasattr(x, "layer_scale"):
            x.layer_scale.data.fill_(val)

class StreamingConvnext(BaseModel):
    model_choices = {"convnext_tiny": convnext_tiny, "convnext_small": convnext_small}

    def __init__(
        self,
        model_name: str,
        tile_size: int,
        loss_fn: torch.nn.functional,
        train_streaming_layers: bool = True,
        use_streaming: bool = True,
        *args,
        **kwargs,
    ):
        assert model_name in list(StreamingConvnext.model_choices.keys())

        self.model_name = model_name

        network = StreamingConvnext.model_choices[model_name](weights="IMAGENET1K_V1")
        stream_net, head = network.features, torch.nn.Sequential(network.avgpool, network.classifier)
        stream_net = torch.nn.Sequential(stream_net[0], stream_net[1])

        # Temporarily replace layernorm, linear, and gelu to not meddle with statistics calculations in streaming
        convert_to_identity(stream_net, torch.nn.LayerNorm)
        convert_to_identity(stream_net, torch.nn.Linear)
        convert_to_identity(stream_net, torch.nn.GELU)

        # Do not use stochastic depth when initializing streaming
        stream_net = freeze_stochastic_depth(stream_net)

        # Set layer scale parameters to one to not meddle with streaming statistics calculations
        set_layer_scale(stream_net, 1.0)

        super().__init__(
            stream_net,
            head,
            tile_size,
            loss_fn,
            train_streaming_layers=train_streaming_layers,
            use_streaming=use_streaming,
            *args,
            **kwargs,
        )

        # check self.stream_network, and reload the proper weights
        self.restore_model_layers()

    def restore_model_layers(self):
        temp_model = StreamingConvnext.model_choices[self.model_name](weights="IMAGENET1K_V1").features
        StreamingConvnext.restore_layers(temp_model, self.stream_network.stream_module)

        #restore_layer_scale(temp_model, self.stream_network.stream_module)


    @staticmethod
    def restore_layers(model_ref, model_rep):
        for ref, rep in zip(
            model_ref.named_children(), model_rep.named_children()
        ):

            n_ref, module_ref = ref
            n_rep, module_rep = rep

            if len(list(module_ref.children())) > 0:
                ## compound module, go inside it
                StreamingConvnext.restore_layers(module_ref, module_rep)

            if isinstance(module_rep, torch.nn.Identity):
                ## simple module
                try:
                    n_ref = int(n_ref)
                    model_rep[n_rep] = model_ref[n_ref]
                except:
                    setattr(model_rep, n_rep, model_ref[int(n_ref)])


if __name__ == "__main__":
    print(torch.cuda.is_available())
    model = StreamingConvnext("convnext_tiny", 1600, nn.MSELoss)
