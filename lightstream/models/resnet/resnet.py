import torch
import torch.nn as nn
from torch.nn import Sequential
from pathlib import Path
from torchvision.models import resnet18, resnet34, resnet50
from lightstream.modules.streaming import StreamingModule


def split_resnet(net, remove_last_block: bool = False):
    """Split resnet architectures into streamable models

    Parameters
    ----------
    net: torch model
        A ResNet model in the format provided by torchvision

    Returns
    -------
    stream_net : torch.nn.Sequential
        The CNN core of the ResNet

    """

    if remove_last_block:
        stream_net = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
    else:
        stream_net = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
        )

    return stream_net


class StreamingResNet(StreamingModule):
    def __init__(
        self,
        encoder: str,
        tile_size: int,
        additional_modules: nn.Module | None = None,
        remove_last_block=False,
        verbose: bool = True,
        deterministic: bool = True,
        saliency: bool = False,
        copy_to_gpu: bool = False,
        statistics_on_cpu: bool = True,
        normalize_on_gpu: bool = True,
        mean: list | None = None,
        std: list | None = None,
        tile_cache_path=None,
    ):
        model_choices = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
        }

        if encoder not in model_choices:
            raise ValueError(f"Invalid model name '{encoder}'. " f"Choose one of: {', '.join(model_choices.keys())}")

        resnet = model_choices[encoder](weights="DEFAULT")

        if additional_modules is not None:
            stream_network = Sequential(
                split_resnet(resnet, remove_last_block=remove_last_block),
                additional_modules,
            )
        else:
            stream_network = split_resnet(resnet, remove_last_block=remove_last_block)

        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        if tile_cache_path is None:
            tile_cache_path = Path.cwd() / Path(f"{encoder}_tile_cache_1_3_{str(tile_size)}_{str(tile_size)}")

        super().__init__(
            stream_network,
            tile_size,
            tile_cache_path,
            verbose=verbose,
            deterministic=deterministic,
            saliency=saliency,
            copy_to_gpu=copy_to_gpu,
            statistics_on_cpu=statistics_on_cpu,
            normalize_on_gpu=normalize_on_gpu,
            mean=mean,
            std=std,
            add_keep_modules=[nn.BatchNorm2d],
        )


if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 4160, 4160)).to("cuda")
    network = StreamingResNet(
        "resnet18",
        4800,
        additional_modules=torch.nn.MaxPool2d((2, 2)),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
    )
    network.to("cuda")
    network.stream_network.device = torch.device("cuda")

    network.stream_network.mean = network.stream_network.mean.to("cuda")
    network.stream_network.std = network.stream_network.std.to("cuda")

    out_streaming = network(img)
    network.stream_network.disable()
    normal_net = network.stream_network.stream_module
    out_normal = normal_net(img)
    diff = out_streaming - out_normal
    print(diff.max())
