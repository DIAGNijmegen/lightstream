import torch

from pathlib import Path

from lightstream.modules.streaming import StreamingModule
from lightstream.models.inceptionnext.model import inceptionnext_atto, inceptionnext_tiny

# input 320x320x3 on float32, torchinfo
# resnet 34     : Forward/backward pass size (MB): 4286.28
# resnet 50     : Forward/backward pass size (MB): 5806.62
# convnext  tiny: Forward/backward pass size (MB): 4286.28
# inception atto: Forward/backward pass size (MB): 1194.36
# inception tiny: Forward/backward pass size (MB): 3907.07

def _set_layer_gamma(model, val=1.0):
    for x in model.modules():
        if hasattr(x, "gamma"):
            x.gamma.data.fill_(val)

class StreamingInceptionNext(StreamingModule):

    model_choices = {"inceptionnext-atto": inceptionnext_atto, "inceptionnext-tiny": inceptionnext_tiny}

    def __init__(
        self,
        model_name: str,
        tile_size: int,
        tile_cache_path: Path | None = None,
        **kwargs
    ):
        assert model_name in list(StreamingInceptionNext.model_choices.keys())
        network = StreamingInceptionNext.model_choices[model_name](pretrained=True)
        stream_network = torch.nn.Sequential(network.stem, network.stages)

        self._get_streaming_options(**kwargs)
        self.streaming_options["before_streaming_init_callbacks"] = [_set_layer_gamma]

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
            "normalize_on_gpu": False,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        self.streaming_options = {**streaming_options, **kwargs}


if __name__ == "__main__":

    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 4160, 4160)).to("cuda")
    network = StreamingInceptionNext(
        "inceptionnext-atto",
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


