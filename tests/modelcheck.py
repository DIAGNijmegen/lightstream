"""
This helper class extends the StreamingModule and checks if streaming is equivalent to non-streaming

It's used for running sanity checks on the StreamingModule class, as well as showing equivalance between streaming models
and non-streaming counterparts.

Only models with a 'simple' design are supported, primarily aimed at torchvision models. This means:

 - The encoder/backbone is a regular CNN that satisfies the assumptions for streaming (no dense layers, normalization can be frozen)
 - The head of the network, most commonly dubbed as net.fc, has a single tensor output, e.g. the logits. multiple outputs are not supported
 - The full model only accepts images as input in a (H,W,3) tensor format

"""

import torch
from lightstream.modules.streaming import StreamingModule
from lightstream.core.scnn import StreamingConv2d
from torchvision.models import resnet18, resnet34, resnet50
from models.resnet.resnet import split_resnet

from models.convnext.convnext import (
    _save_parameters,
    _prepare_for_streaming_statistics,
    _toggle_stochastic_depth,
    _restore_layers,
    _set_layer_scale,
    _convert_to_identity,
)
from torchvision.models import convnext_tiny


def create_dummy_data(self):
    img_size = 1600
    image = torch.FloatTensor(3, img_size, img_size).normal_(0, 1)
    image = image.type(self.dtype)
    image = image.to(self.device)

    target = torch.tensor(50.0)  # large value so we get larger gradients
    target = target.type(self.dtype)
    target = target.to(self.device)

    return image, target


class ModelCheck(StreamingModule):
    def __init__(self, stream_network, tile_size, loss_fn, *args, **kwargs):
        super().__init__(stream_network, tile_size, *args, **kwargs)
        self.loss_fn = loss_fn

    def _remove_streaming_network(self):
        """Converts the streaming network into a non-streaming network

        The former streaming encoder can be addressed as self.stream_network

        """

        # Convert streamingConv2D into regular Conv2D and turn off streaming hooks
        self.disable_streaming()
        temp = self.stream_network.stream_module

        # torch modules cannot be overridden normally, so delete and reassign
        del self.stream_network
        self.stream_network = temp

    def forward(self, x):
        if self.use_streaming:
            return self.forward_streaming(x)

        return self.stream_network(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        image, target = batch
        self.image = image

        self.str_output = self.forward(image)

        if self.use_streaming:
            self.str_output.requires_grad = True

        out = torch.mean(self.str_output)
        loss = self.loss_fn(out, target)

        # Can be a dict as long as it has a "loss" key
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step
        return {"loss": loss, "forward_output": out}

    def backward(self, loss):
        loss.backward()

        if self.train_streaming_layers and self.use_streaming:
            self.backward_streaming(self.image, self.str_output.grad)
        del self.str_output

    def step_once(self, batch):
        # saliency is computed within scnn, but not for conventional training
        if not self.use_streaming:
            batch[0].requires_grad = True

        outputs = self.training_step(batch, 0)
        self.backward(outputs["loss"])

        return outputs

    def gather_output(self, outputs):
        if self.use_streaming:
            saliency = self.stream_network.saliency_map.detach().cpu().numpy()
            kernel_grads = self.gather_kernel_gradients(StreamingConv2d)
        else:
            saliency = self.image.grad.detach().cpu().numpy()
            kernel_grads = self.gather_kernel_gradients(torch.nn.Conv2d)

        return {
            "loss": outputs["loss"].detach().cpu().numpy(),
            "forward_output": outputs["forward_output"].detach().cpu().numpy(),
            "input_gradient": saliency,
            "kernel_gradients": kernel_grads,
        }

    def to_device(self):
        self.stream_network.device = self.device
        self.stream_network.mean = self.mean
        self.stream_network.std = self.std
        self.stream_network.dtype = self.dtype

    def run(self, batch):
        # run once with streaming and gather output
        self.to_device()
        streaming_step_results = self.step_once(batch)
        streaming_stats = self.gather_output(streaming_step_results)

        # Remove streaming entirely, and reset gradients
        self._remove_streaming_network()
        self.stream_network.zero_grad()

        self.to_device()
        step_results = self.step_once(batch)
        normal_stats = self.gather_output(step_results)

        return streaming_stats, normal_stats

    def gather_kernel_gradients(self, module):
        """Gather the kernel gradient for the specified module"""

        kernel_gradients = []

        # stream_network can be used for both streaming and non-streaming
        # the only difference is Conv2D layers are turned into streamingConv2D layers

        model = self.stream_network.stream_module if self.use_streaming else self.stream_network

        for i, layer in enumerate(model.modules()):
            if isinstance(layer, module):
                if layer.weight.grad is not None:
                    kernel_gradients.append(layer.weight.grad.clone().detach().cpu().numpy())

        return kernel_gradients


class ModelCheckConvNext(StreamingModule):
    model_choices = {"convnext_tiny": convnext_tiny}

    def __init__(
        self,
        model_name: str,
        tile_size: int,
        loss_fn: torch.nn.functional,
        train_streaming_layers: bool = True,
        use_streaming: bool = True,
        use_stochastic_depth: bool = True,
        *args,
        **kwargs,
    ):
        assert model_name in list(ModelCheckConvNext.model_choices.keys())

        self.model_name = model_name
        self.use_stochastic_depth = use_stochastic_depth

        network = ModelCheckConvNext.model_choices[model_name](weights="IMAGENET1K_V1")
        stream_network, head = network.features, torch.nn.Sequential(network.avgpool, network.classifier)

        # Save parameters for easy recovery of module parameters later
        state_dict = _save_parameters(stream_network)

        # Prepare for streaming tile statistics calculations
        _prepare_for_streaming_statistics(stream_network)

        super().__init__(stream_network, tile_size, copy_to_gpu=False, saliency=True, verbose=False)

        # check self.stream_network, and reload the proper weights
        self._restore_model_layers()

        # re apply layer scale weights and stochastic depth settings
        self.stream_network.stream_module.load_state_dict(state_dict)

        if use_stochastic_depth:
            _toggle_stochastic_depth(stream_network, training=True)

        self.loss_fn = loss_fn

    def _restore_model_layers(self):
        temp_model = ModelCheckConvNext.model_choices[self.model_name](weights="IMAGENET1K_V1").features
        _restore_layers(temp_model, self.stream_network.stream_module)

    def _remove_streaming_network(self):
        """Converts the streaming network into a non-streaming network

        The former streaming encoder can be addressed as self.stream_network

        """

        # Convert streamingConv2D into regular Conv2D and turn off streaming hooks
        self.disable_streaming()
        temp = self.stream_network.stream_module

        # torch modules cannot be overridden normally, so delete and reassign
        del self.stream_network
        self.stream_network = temp

    def forward(self, x):
        if self.use_streaming:
            return self.forward_streaming(x)

        return self.stream_network(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        image, target = batch
        self.image = image

        self.str_output = self.forward(image)

        if self.use_streaming:
            self.str_output.requires_grad = True

        out = torch.mean(self.str_output)
        loss = self.loss_fn(out, target)

        # Can be a dict as long as it has a "loss" key
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step
        return {"loss": loss, "forward_output": out}

    def backward(self, loss):
        loss.backward()

        if self.train_streaming_layers and self.use_streaming:
            _toggle_stochastic_depth(self.stream_network.stream_module, training=False)
            self.backward_streaming(self.image, self.str_output.grad)

        #print("MODEL DEBUG", self.stream_network.stream_module[1][0].stochastic_depth.training)
        del self.str_output

    def step_once(self, batch):
        # saliency is computed within scnn, but not for conventional training
        if not self.use_streaming:
            batch[0].requires_grad = True

        outputs = self.training_step(batch, 0)
        self.backward(outputs["loss"])

        return outputs

    def gather_output(self, outputs):
        if self.use_streaming:
            saliency = self.stream_network.saliency_map.detach().cpu().numpy()
            kernel_grads = self.gather_kernel_gradients(StreamingConv2d)
        else:
            saliency = self.image.grad.detach().cpu().numpy()
            kernel_grads = self.gather_kernel_gradients(torch.nn.Conv2d)

        return {
            "loss": outputs["loss"].detach().cpu().numpy(),
            "forward_output": outputs["forward_output"].detach().cpu().numpy(),
            "input_gradient": saliency,
            "kernel_gradients": kernel_grads,
        }

    def to_device(self):
        self.stream_network.device = self.device
        self.stream_network.mean = self.mean
        self.stream_network.std = self.std
        self.stream_network.dtype = self.dtype

    def run(self, batch):
        # run once with streaming and gather output
        self.to_device()
        streaming_step_results = self.step_once(batch)
        streaming_stats = self.gather_output(streaming_step_results)

        # Remove streaming entirely, and reset gradients
        self._remove_streaming_network()
        self.stream_network.zero_grad()

        self.to_device()
        step_results = self.step_once(batch)
        normal_stats = self.gather_output(step_results)

        return streaming_stats, normal_stats

    def gather_kernel_gradients(self, module):
        """Gather the kernel gradient for the specified module"""

        kernel_gradients = []

        # stream_network can be used for both streaming and non-streaming
        # the only difference is Conv2D layers are turned into streamingConv2D layers

        model = self.stream_network.stream_module if self.use_streaming else self.stream_network

        for i, layer in enumerate(model.modules()):
            if isinstance(layer, module):
                if layer.weight.grad is not None:
                    kernel_gradients.append(layer.weight.grad.clone().detach().cpu().numpy())

        return kernel_gradients


if __name__ == "__main__":
    img_size = 1600 + 320
    tile_size = 1600

    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = torch.FloatTensor(3, img_size, img_size).normal_(0, 1)
    image = image.type(dtype).to(device)

    target = torch.tensor(50.0)  # large value so we get larger gradients
    target = target.type(dtype).to(device)

    batch = (image[None], target)

    model = resnet18()
    stream_net, head = split_resnet(model)

    model_check = ModelCheck(stream_net, tile_size, loss_fn=torch.nn.MSELoss(), verbose=False, saliency=True)
    model_check.to(device)
    model_check.to(dtype)
    streaming, normal = model_check.run(batch)
