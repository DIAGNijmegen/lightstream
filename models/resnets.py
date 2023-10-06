from typing import Any

import torch
import torch.nn as nn

from src.stream import StreamingModule
from torchvision.models import resnet18, resnet34, resnet50

class StreamingResNet(StreamingModule):
    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

    def __init__(self, model_name, tile_size, *args, **kwargs):

        assert model_name in list(StreamingResNet.model_choices.keys())
        network = StreamingResNet.model_choices[model_name](weights="IMAGENET1K_V1")
        super().__init__(network, tile_size, train_streaming_layers=False)

    def split_model(self, model):
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



if __name__ == "__main__":
    model = StreamingResNet("resnet18", tile_size=1600)
    model.eval()
    inputs = torch.ones((1, 3, 3200, 3200))

    output_start = model(inputs)

    test_model = resnet18(weights="IMAGENET1K_V1")
    test_model.eval()
    output_test = test_model(inputs)

    print(torch.sum(output_test - output_start))


    # Check backward loop