import torch
import torch.nn as nn
from torch.nn import Sequential
from pathlib import Path

from lightstream.modules.lightningstreaming import StreamingModule
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
    def __init__(
        self,
        encoder: str,
        tile_size: int,
        additional_modules: nn.Module=None,
        remove_last_block: bool = False,
        use_stochastic_depth: bool = False,
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
        model_choices = {"inceptionnext-atto": inceptionnext_atto, "inceptionnext-tiny": inceptionnext_tiny}
        self.use_stochastic_depth = use_stochastic_depth

        if encoder not in model_choices:
            raise ValueError(f"Invalid model name '{encoder}'. " f"Choose one of: {', '.join(model_choices.keys())}")

        network = model_choices[encoder](weights="DEFAULT")

        end = 4
        if remove_last_block:
            end = 3

        if additional_modules is not None:
            stream_network = Sequential(network.stem, network.stages[0:end], additional_modules)
        else:
            stream_network = Sequential(network.stem, network.stages[0:end])


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
            before_streaming_init_callbacks=[_set_layer_gamma],
        )


if __name__ == "__main__":

    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 4160, 4160)).to("cuda")
    network = StreamingInceptionNext(
        "inceptionnext-tiny",
        6400,
        remove_last_block=False,
        additional_modules=torch.nn.MaxPool2d(2,2),
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
