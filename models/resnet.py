import torch
import torch.nn as nn
from lightstream.scnn import StreamingConv2d


from modules.base import BaseModel
from torchvision.models import resnet18, resnet34, resnet50


def split_resnet(net):
    stream_net = nn.Sequential(
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4,
    )
    head = nn.Sequential(net.avgpool, nn.Flatten(), net.fc)
    return stream_net, head


class StreamingResNet(BaseModel):
    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

    def __init__(
        self,
        model_name: str,
        tile_size,
        loss_fn,
        train_streaming_layers=True,
        use_streaming=True,
        *args,
        **kwargs
    ):
        assert model_name in list(StreamingResNet.model_choices.keys())
        network = StreamingResNet.model_choices[model_name](weights="IMAGENET1K_V1")
        stream_net, head = split_resnet(network)
        super().__init__(
            stream_net,
            head,
            tile_size,
            loss_fn,
            train_streaming_layers=train_streaming_layers,
            use_streaming=use_streaming,
            *args,
            **kwargs
        )


if __name__ == "__main__":
    # Batchnorm can have non-empty grads, conv2d layers should be none after mod.eval()

    # Any input that plays nicely with the output stride, i.e. multiples of 32
    img_size = 2048
    model = StreamingResNet(
        "resnet18",
        tile_size=1504,
        loss_fn=torch.nn.MSELoss(),
        train_streaming_layers=True,
        use_streaming=True,
        saliency=True,
        gather_input_gradients=True,
    )

    dtype = torch.double  # test with double precision
    model.stream_network.stream_module.type(dtype)
    model.stream_network.stream_module.cuda()

    model.head.type(dtype)
    model.head.cuda()

    model.stream_network.device = "cuda"
    model.stream_network.dtype = dtype

    model.freeze_streaming_normalization_layers()

    # %%
    image = torch.FloatTensor(3, img_size, img_size).normal_(0, 1)
    image = image.type(dtype)
    image = image.cuda()

    target = torch.tensor(50.0)  # large value so we get larger gradients
    target = target.type(dtype)
    target = target.cuda()

    fmap_streaming, stream_output, loss = model.training_step((image[None], target), 1)

    loss.backward()
    model.backward_streaming(image[None], fmap_streaming.grad)

    streaming_conv_gradients = []

    for i, layer in enumerate(model.stream_network.stream_module.modules()):
        if isinstance(layer, StreamingConv2d):
            if layer.weight.grad is not None:
                streaming_conv_gradients.append(layer.weight.grad.clone())

    model.disable_streaming()
    model.use_streaming = False

    for i, layer in enumerate(model.stream_network.stream_module.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            print(layer)
            if layer.weight.grad is not None:
                layer.weight.grad.data.zero_()
                layer.bias.grad.data.zero_()

    conventional_gradients = []
    inps = []

    def save_grad(module, grad_in, grad_out):
        global conventional_gradients
        conventional_gradients.append(grad_out[0].clone())

    for i, layer in enumerate(model.stream_network.stream_module.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            layer.register_backward_hook(save_grad)

    image.requires_grad = True
    conventional_fmap, conventional_output, loss = model.training_step(
        (image[None], target), 1
    )

    conventional_output.max()

    max_error = torch.abs(stream_output - conventional_output).max().item()

    if max_error < 1e-7:
        print("Equal output to streaming")
    else:
        print("NOT equal output to streaming"),
        print("error:", max_error)

    loss.backward()
