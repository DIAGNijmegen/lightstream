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

if __name__ == '__main__':
    print(torch.cuda.is_available())