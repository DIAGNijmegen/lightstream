"""
A class to check for equivalance between a streaming network, and it's non-streaming counterpart
the input to the class is the streaming part of the network. Then, we compute forward and backward passes for the
case where streaming is enabled, and using conventional training with the same network.

We then compare the following statistics:
1. The output from the forward pass.
2. Gradients with respect to the input image, which checks for equivalance in the backward input
3. Gradients of the convolutional kernels, which cheks for equivalance between the gradients w.r.t the parameters


In order to compute a backward pass, a simple mse loss over the mena of the final feature maps of the models is computed.
However, it should not matter which loss is used.

"""

import torch
from lightstream.scnn import StreamingCNN, StreamingConv2d


class ModelCheck:
    def __init__(self, stream_network, tile_size=1920, img_size=2048, **kwargs):
        self.tile_size = tile_size
        self.img_size = img_size
        self.loss_fn = torch.nn.MSELoss()

        # Set to double for higher precision
        self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stream_network = stream_network
        self.stream_network.type(self.dtype).to(self.device)
        self.freeze_normalization_layers()

        self.sCNN = StreamingCNN(
            self.stream_network,
            tile_shape=(1, 3, self.tile_size, self.tile_size),
            deterministic=kwargs.get("deterministic", True),
            saliency=kwargs.get("saliency", True),
            gather_gradients=kwargs.get("gather_gradients", True),
            copy_to_gpu=kwargs.get("copy_to_gpu", True),
            verbose=kwargs.get("verbose", True),
        )

        self.streaming_outputs = {}
        self.normal_outputs = {}
        self.image, self.target = self.create_dummy_data()

    def freeze_normalization_layers(self):
        """Normalization layers are not local ops, so freeze them"""
        for mod in self.stream_network.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()

    def gather_outputs(self):
        self.gather_streaming_outputs()
        self.sCNN.disable()
        # Reset grad attributes obtained using streaming
        self.stream_network.zero_grad()
        self.gather_normal_outputs()

    def create_dummy_data(self):
        image = torch.FloatTensor(3, self.img_size, self.img_size).normal_(0, 1)
        image = image.type(self.dtype)
        image = image.to(self.device)

        target = torch.tensor(50.0)  # large value so we get larger gradients
        target = target.type(self.dtype)
        target = target.to(self.device)

        return image, target

    def gather_streaming_outputs(self):
        # saliency is computed within scnn, but not for conventional training

        output = self.forward_model(self.image, use_streaming=True)

        pred = torch.mean(output)
        loss = self.loss_fn(pred, self.target)
        loss.backward()

        self.sCNN.backward(self.image[None], output.grad)

        self.streaming_outputs = {
            "forward_output": pred.detach().cpu().numpy(),
            "input_gradient": self.sCNN.saliency_map.detach().cpu().numpy(),
            "kernel_gradients": self.gather_kernel_gradients(StreamingConv2d),
        }

    def gather_normal_outputs(self):
        # saliency is computed within scnn, but not for conventional training
        self.image.requires_grad = True

        output = self.forward_model(self.image, use_streaming=False)

        pred = torch.mean(output)
        loss = self.loss_fn(pred, self.target)
        loss.backward()

        self.normal_outputs = {
            "forward_output": pred.detach().cpu().numpy(),
            "input_gradient": self.image.grad.detach().cpu().numpy(),
            "kernel_gradients": self.gather_kernel_gradients(torch.nn.Conv2d),
        }

    def forward_model(self, image, use_streaming):
        if use_streaming:
            fmap = self.sCNN(image[None])
            # fmap is leaf tensor when streaming
            fmap.requires_grad = True
        else:
            fmap = self.stream_network(image[None])
        # Create a single point output for the loss artificially
        return fmap

    def gather_kernel_gradients(self, module):
        """Gather the kernel gradient for the specified module"""

        kernel_gradients = []

        # stream_network can be used for both streaming and non-streaming
        # the only difference is Conv2D layers are turned into streamingConv2D layers
        for i, layer in enumerate(self.stream_network.modules()):
            if isinstance(layer, module):
                if layer.weight.grad is not None:
                    kernel_gradients.append(
                        layer.weight.grad.clone().detach().cpu().numpy()
                    )

        return kernel_gradients
