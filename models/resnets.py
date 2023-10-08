import torch
import torch.nn as nn

from modules.streaming import StreamingModule
from streaming.scnn import StreamingConv2d
from torchvision.models import resnet18, resnet34, resnet50


def split_resnet(model):
    stream_net = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    head = nn.Sequential(model.avgpool, nn.Flatten(), model.fc)
    return stream_net, head


class StreamingResNet(StreamingModule):
    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

    def __init__(self, model_name, tile_size, *args, **kwargs):

        assert model_name in list(StreamingResNet.model_choices.keys())
        network = StreamingResNet.model_choices[model_name](weights="IMAGENET1K_V1")
        stream_net, head = split_resnet(network)
        super().__init__(tile_size, stream_net, *args, **kwargs)


if __name__ == "__main__":
    # Batchnorm can have non-empty grads, conv2d layers should be none after mod.eval()

    model = StreamingResNet("resnet18",
                            tile_size=1024,
                            train_streaming_layers=True,
                            enable_streaming=True,
                            saliency=True,
                            gather_input_gradients=True)


    dtype = torch.double  # test with double precision
    model.stream_network.type(dtype)
    model.stream_network.cuda()

    img_size = 1504
    model.eval()

    # %%
    image = torch.FloatTensor(3, img_size, img_size).normal_(0, 1)
    target = torch.tensor(50.)  # large value so we get larger gradients

    image = image.type(dtype)
    target = target.type(dtype)

    target = target.cuda()
    image = image.cuda()


    output_start = model(image[None])

    batch = (image, target)
    model.train()

    model.freeze_normalization_layers()

    fmap,stream_output, loss = model.training_step(batch, 1)

    fmap_stream = fmap.detach().cpu().numpy()

    model.backward(loss, batch, fmap, stream_output)
    fmap_stream_grad = fmap.grad.detach()

    streaming_conv_gradients = []

    for i, layer in enumerate(model.stream_network.modules()):
        if isinstance(layer, StreamingConv2d):
            if layer.weight.grad is not None:
                streaming_conv_gradients.append(layer.weight.grad.clone())

    print("sal shape", model.sCNN.saliency_map.shape)

    model.sCNN.disable()

    print("break")




    def split_model(model):
        stream_net = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        head = nn.Sequential(model.avgpool, nn.Flatten(), model.fc)
        return stream_net, head




    test_model = resnet18(weights="IMAGENET1K_V1")

    stream_net, head = split_model(test_model)

    dtype = torch.double  # test with double precision
    stream_net.type(dtype)
    stream_net.cuda()



    # freeze bn statistics
    for x in stream_net.modules():
        if isinstance(x, torch.nn.BatchNorm2d):
            x.eval()

    conventional_gradients = []
    inps = []


    def save_grad(module, grad_in, grad_out):
        global conventional_gradients
        conventional_gradients.append(grad_out[0].clone())


    for i, layer in enumerate(stream_net.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            layer.register_full_backward_hook(save_grad)

    image.requires_grad = True
    fmap_non_streaming = stream_net(image[None])
    fmap_non_streaming.retain_grad()

    conventional_output = torch.sigmoid(torch.mean(fmap_non_streaming))
    conventional_output.max()

    # In that case this check may fail.
    max_error = torch.abs(stream_output - conventional_output).max().item()

    if max_error < 1e-7:
        print("Equal output to streaming")
    else:
        print("NOT equal output to streaming"),
        print("error:", max_error)

    criterion = torch.nn.MSELoss()

    loss = criterion(conventional_output, target)
    print(loss.dtype)
    loss.backward()
    print(loss.dtype)
    print(conventional_gradients[-1].shape)

    diff = image.grad.detach().cpu().numpy() - model.sCNN.saliency_map[0].numpy()
    print(diff.max())


    ## Compare gradients of conv kernels
    normal_conv_gradients = []
    j = 0
    for i, layer in enumerate(stream_net.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            if layer.weight.grad is not None:
                normal_conv_gradients.append(layer.weight.grad)
                print('Conv layer', j, '\t', layer)
                j += 1



    print('Streaming', '\n')
    for i in range(len(streaming_conv_gradients)):
        print("Conv layer", i, "\t average gradient size:",
              float(torch.mean(torch.abs(streaming_conv_gradients[i].data))))

    print('Conventional', '\n')
    for i in range(len(normal_conv_gradients)):
        print("Conv layer", i, "\t average gradient size:",
              float(torch.mean(torch.abs(normal_conv_gradients[i].data))))


    for i in range(len(streaming_conv_gradients)):
        diff = torch.abs(streaming_conv_gradients[i].data - normal_conv_gradients[i].data)
        max_diff = diff.max()
        print("Conv layer", i, "\t max difference between kernel gradients:",
              float(max_diff))
              
