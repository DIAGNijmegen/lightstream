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

from copy import deepcopy


class StreamingConstructor:
    def __init__(
        self,
        model,
        before_streaming_init_callbacks: list | None = None,
        after_streaming_init_callbacks: list | None = None,
        *args,
        **kwargs
    ):
        self.model = model
        self.model_copy = deepcopy(self.model)

        # Save parameters for easy recovery of module parameters later
        self.state_dict = self.save_parameters()

        self.keep_modules = (
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.AvgPool1d,
            torch.nn.AvgPool2d,
            torch.nn.AvgPool3d,
            torch.nn.MaxPool1d,
            torch.nn.MaxPool2d,
            torch.nn.MaxPool3d,
        )

        self.before_streaming_init_callbacks = before_streaming_init_callbacks
        self.after_streaming_init_callbacks = after_streaming_init_callbacks

    def create_streaming_model(self):
        # Prepare for streaming tile statistics calculations

        self.prepare_for_streaming_statistics()

        # Call scnn code here

        # ....

        # ....

        # end scnn code here

        # check self.stream_network, and reload the proper weights
        self.restore_model_layers(self.model_copy, self.model)

        # re apply layer scale weights and stochastic depth settings
        self.stream_network.stream_module.load_state_dict(self.state_dict)

    def save_parameters(self):
        state_dict = self.model.state_dict()
        state_dict = deepcopy(state_dict)
        return state_dict

    def convert_to_identity(self, model: torch.nn.modules, old: torch.nn.modules):
        """Convert non-conv and non-local pooling layers to identity

        Parameters
        ----------
        model : torch.nn.Sequential
            The model to substitute
        old :

        """
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                # compound module, go inside it
                self.convert_to_identity(module, old)

            # if new module is assigned to a variable, e.g. new = nn.Identity(), then it's considered a duplicate in
            # module.named_children used later. Instead, we use in-place assignment, so each new module is unique
            if not isinstance(module, old):
                # simple module
                try:
                    n = int(n)
                    model[n] = torch.nn.Identity()
                except:
                    setattr(model, str(n), torch.nn.Identity())

    def prepare_for_streaming_statistics(self):
        # Temporarily replace layernorm, linear, and gelu to not meddle with statistics calculations in streaming

        self.convert_to_identity(self.model, self.keep_modules)

        for fun in self.before_streaming_init_callbacks:
            fun()

    def restore_model_layers(self, model_ref, model_rep):
        for ref, rep in zip(model_ref.named_children(), model_rep.named_children()):
            n_ref, module_ref = ref
            n_rep, module_rep = rep

            if len(list(module_ref.children())) > 0:
                # compound module, go inside it
                self.restore_model_layers(module_ref, module_rep)

            if isinstance(module_rep, torch.nn.Identity):
                # simple module
                try:
                    n_ref = int(n_ref)
                    model_rep[n_rep] = model_ref[n_ref]
                except:
                    setattr(model_rep, n_rep, model_ref[int(n_ref)])
