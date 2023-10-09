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
import numpy as np

from stream.scnn import StreamingCNN, StreamingConv2d


class Equivalence:
    def __init__(self, stream_network, tile_size=1504, img_size=2048, **kwargs):
        self.tile_size = tile_size
        self.img_size = img_size
        self.loss_fn = torch.nn.MSELoss()

        # Set to double for higher precision
        self.dtype = torch.double
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stream_network = stream_network
        self.stream_network.to(self.device)
        self.freeze_normalization_layers()

        self.sCNN = StreamingCNN(
            self.stream_network,
            tile_shape=(1, 3, self.tile_size, self.tile_size),
            deterministic=kwargs.get("deterministic", True),
            saliency=kwargs.get("saliency", False),
            gather_gradients=kwargs.get("gather_gradients", False),
            copy_to_gpu=kwargs.get("copy_to_gpu", True),
            verbose=kwargs.get("verbose", True),
        )


    def freeze_normalization_layers(self):
        """ Normalization layers are not local ops, so freeze them """
        for mod in self.stream_network.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()


    def run_tests(self):
        stream_output, stream_saliency, streaming_gradients = self.run_model(use_streaming=True)
        self.sCNN.disable()

        # Reset grad attributes obtained using streaming
        self.stream_network.zero_grad()

        conventional_output, input_saliency, conventional_gradients = self.run_model(use_streaming = False)

        self.test_forward_output(stream_output, conventional_output)
        self.test_input_gradients(stream_saliency, input_saliency)
        self.test_kernel_gradients(streaming_gradients, conventional_gradients)

    def run_model(self, use_streaming=False):
        image, target = self.create_dummy_data()
        # saliency is computed within scnn, but not for conventional training
        image.requires_grad = True if not use_streaming else False

        output = self.forward_model(use_streaming)

        loss = self.loss_fn(output, target)
        loss.backward()

        if use_streaming:
            self.sCNN.backward(image[None], stream_output.grad)
            input_gradients = self.sCNN.saliency_map.detach().cpu().numpy()
            kernel_gradients = self.gather_kernel_gradients(StreamingConv2d)
        else:
            input_gradients = image.grad.detach().cpu().numpy()
            kernel_gradients = self.gather_kernel_gradients(torch.nn.Conv2d)

        return output, input_gradients, kernel_gradients


    def forward_model(self, use_streaming):
        if use_streaming:
            fmap = self.sCNN(image[None])
            # fmap is leaf tensor when streaming
            fmap.requires_grad = True
        else:
            fmap = self.stream_network(image[None])
        # Create a single point output for the loss artificially
        return torch.mean(fmap)


    def create_dummy_data(self):
        image = torch.FloatTensor(3, self.img_size, self.img_size).normal_(0, 1)
        image = image.type(dtype)
        image = image.to(self.device)

        target = torch.tensor(50.)  # large value so we get larger gradients
        target = target.type(dtype)
        target = target.to(self.device)

        return image, target


    def gather_kernel_gradients(self, module):
        """ Gather the kernel gradient for the specified module """

        kernel_gradients = []

        for i, layer in enumerate(self.stream_network.modules()):
            if isinstance(layer, module):
                if layer.weight.grad is not None:
                    kernel_gradients.append(layer.weight.grad.clone())

        return kernel_gradients

    def get_kernel_sizes(self, kernel_gradients):
        for i in range(len(kernel_gradients)):
            print("Conv layer", i, "\t average gradient size:",
                  float(torch.mean(torch.abs(kernel_gradients[i].cpu().numpy()))))

    def test_forward_output(self):
        max_error = torch.abs(stream_output - conventional_output).max().item()

        if max_error < 1e-7:
            print("Equal output to streaming")
        else:
            print("NOT equal output to streaming"),
            print("error:", max_error)
    def test_input_gradients(self, stream_saliency, input_saliency):
        diff = np.abs(input_saliency().cpu().numpy() - stream_saliency[0].numpy())
        print(diff)


    def test_kernel_gradients(self, streaming_kernel_gradients, conventional_kernel_gradients):
        for i in range(len(streaming_conv_gradients)):
            diff = torch.abs(streaming_conv_gradients[i].data - normal_conv_gradients[i].data)
            max_diff = diff.max()
            print("Conv layer", i, "\t max difference between kernel gradients:",
                  float(max_diff))


if __name__ == "__main__":
    print("A cat says meow when he's hungry")