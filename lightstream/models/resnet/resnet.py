import torch
import torch.nn as nn
from lightstream.modules.base import BaseModel
from torchvision.models import resnet18, resnet34, resnet50

def split_resnet(net, num_classes=1000):
    """ Split resnet architectures into backbone and fc modules

    The stream_net will contain the CNN backbone that is capable for streaming.
    The fc model is not streamable and will be a separate module
    If num_classes are specified as a kwarg, then a new fc model will be created with the desired classes

    Parameters
    ----------
    net: torch model
        A ResNet model in the format provided by torchvision
    kwargs

    Returns
    -------
    stream_net : torch.nn.Sequential
        The CNN backbone of the ResNet
    head : torch.nn.Sequential
        The head of the model, defaults to the fc model provided by torchvision.

    """

    stream_net = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
    )

    # 1000 classes is the default from ImageNet classification
    if num_classes != 1000:
        net.fc = torch.nn.Linear(512, num_classes)
        torch.nn.init.xavier_normal_(net.fc.weight)
        net.fc.bias.data.fill_(0)  # type:ignore

    head = nn.Sequential(net.avgpool, nn.Flatten(), net.fc)

    return stream_net, head


class StreamingResNet(BaseModel):
    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

    def __init__(
        self,
        model_name: str,
        tile_size: int,
        loss_fn: torch.nn.functional,
        train_streaming_layers: bool = True,
        use_streaming: bool = True,
        *args,
        **kwargs
    ):
        assert model_name in list(StreamingResNet.model_choices.keys())
        network = StreamingResNet.model_choices[model_name](weights="IMAGENET1K_V1")
        stream_network, head = split_resnet(network, num_classes=kwargs.get("num_classes", 1000))
        super().__init__(
            stream_network,
            head,
            tile_size,
            loss_fn,
            train_streaming_layers=train_streaming_layers,
            use_streaming=use_streaming,
            *args,
            **kwargs
        )


if __name__ == "__main__":
    print(torch.cuda.is_available())
    model = StreamingResNet("resnet18", 1600, nn.MSELoss)
