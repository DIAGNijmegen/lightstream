import torch
import torch.nn as nn
from pathlib import Path
from lightstream.modules.streaming import StreamingModule
from torchvision.models import resnet18, resnet34, resnet50

def split_resnet(net, encoder: str):
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

    if encoder == "resnet39":
        stream_net = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
    else:
        stream_net = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)

    return stream_net


class StreamingResNet(StreamingModule):
    # Resnet  minimal tile size based on tile statistics calculations:
    # resnet18 : 960

    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet39": resnet50, "resnet50": resnet50}

    def __init__(
        self,
        model_name: str,
        tile_size: int,
        tile_cache_path: Path | None = None,
        **kwargs
    ):
        assert model_name in list(StreamingResNet.model_choices.keys())
        network = StreamingResNet.model_choices[model_name](weights="DEFAULT")
        stream_network = split_resnet(network, encoder=model_name)

        self._get_streaming_options(**kwargs)
        self.streaming_options["add_keep_modules"] = [torch.nn.BatchNorm2d]

        if tile_cache_path is None:
            tile_cache_path = Path.cwd() / Path(f"{model_name}_tile_cache_1_3_{str(tile_size)}_{str(tile_size)}")

        super().__init__(
            stream_network,
            tile_size,
            tile_cache_path=tile_cache_path,
            **self.streaming_options,
        )

    def _get_streaming_options(self, **kwargs):
        """Set streaming defaults, but overwrite them with values of kwargs if present."""

        # We need to add torch.nn.Batchnorm to the keep modules, because of some in-place ops error if we don't
        # https://discuss.pytorch.org/t/register-full-backward-hook-for-residual-connection/146850
        streaming_options = {
            "verbose": True,
            "copy_to_gpu": False,
            "statistics_on_cpu": True,
            "normalize_on_gpu": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        self.streaming_options = {**streaming_options, **kwargs}



if __name__ == "__main__":
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 4160, 4160)).to("cuda")
    network = StreamingResNet(
        "resnet39",
        4800,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        normalize_on_gpu=False,
    )
    network.to("cuda")
    network.stream_network.device = torch.device("cuda")

    network.stream_network.mean = network.stream_network.mean.to("cuda")
    network.stream_network.std = network.stream_network.std.to("cuda")

    out_streaming = network(img)

    network.disable_streaming_hooks()
    normal_net = network.stream_network.stream_module
    out_normal = normal_net(img)
    diff = out_streaming - out_normal
    print(diff.max())
