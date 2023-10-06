from src.scnn import StreamingCNN
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)


class TestResNet:
    def __init__(self):
        pass

    def test_resnet18(self):
        pass

    def test_resnet34(self):
        pass

    def test_resnet50(self):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)


def test_model_logic():
    tile_size = 1600
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    stream_net, net = split_model(model)
    stream_net = StreamingCNN(
        stream_net,
        tile_shape=(1, 3, tile_size, tile_size),
        deterministic=True,
        saliency=False,
        gather_gradients=False,
        copy_to_gpu=True,
        verbose=True,
    )

    model_input = torch.ones(1, 3, 3200, 3200)
    first_out = stream_net.forward(model_input)
    out_streaming = net(first_out)

    out_normal = model(model_input)

    print(torch.sum(out_streaming - out_normal))


# Only stream the conv layers, do not include the heads.
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
    net = nn.Sequential(model.avgpool, nn.Flatten(), model.fc)
    return stream_net, net


def freeze_bn_layers(module):
    freeze_layers = [l.eval() for l in module.layers if isinstance(l, nn.BatchNorm2D)]


def test_pytest():
    assert 1 == 1
