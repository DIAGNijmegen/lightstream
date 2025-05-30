from lightstream.modules.lightningstreaming import StreamingModule
from torchvision.ops import StochasticDepth
from torchvision.models.convnext import convnext_tiny, convnext_small
from pathlib import Path

def _toggle_stochastic_depth(model, training=False):
    for m in model.modules():
        if isinstance(m, StochasticDepth):
            m.training = training


def _set_layer_scale(model, val=1.0):
    for x in model.modules():
        if hasattr(x, "layer_scale"):
            x.layer_scale.data.fill_(val)



class StreamingConvnext(StreamingModule):
    model_choices = {"convnext-tiny": convnext_tiny, "convnext-small": convnext_small}

    def __init__(
            self,
            model_name: str,
            tile_size: int,
            use_stochastic_depth: bool = False,
            tile_cache_path: str = None,
            **kwargs,
    ):
        assert model_name in list(StreamingConvnext.model_choices.keys())

        self.model_name = model_name
        self.use_stochastic_depth = use_stochastic_depth

        network = StreamingConvnext.model_choices[model_name](weights="DEFAULT")
        self._get_streaming_options(**kwargs)

        # Set these here so that they are no accidentally overwritten by the user
        # These functions are necessary to calculate tile statistics correctly
        self.streaming_options["before_streaming_init_callbacks"] = [_set_layer_scale]
        self.streaming_options["after_streaming_init_callbacks"] = [_toggle_stochastic_depth]

        if tile_cache_path is None:
            tile_cache_path = Path.cwd() / Path(f"{model_name}_tile_cache_1_3_{str(tile_size)}_{str(tile_size)}")


        super().__init__(
            network.features,
            tile_size,
            tile_cache_path=tile_cache_path,
            **self.streaming_options,
        )

        # By default, the after_streaming_init callback turns sd off
        _toggle_stochastic_depth(self.stream_network.stream_module, training=self.use_stochastic_depth)

    def _get_streaming_options(self, **kwargs):
        """Set streaming defaults, but overwrite them with values of kwargs if present."""

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
    import torch
    print(" is cuda available? ", torch.cuda.is_available())
    img = torch.rand((1, 3, 4160, 4160)).to("cuda")
    network = StreamingConvnext(
        "convnext-tiny",
        4800,
        use_stochastic_depth=False,
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
