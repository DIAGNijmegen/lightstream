from lightstream.models.inceptionnext.inceptionnext import inceptionnext_atto, inceptionnext_tiny
from torchinfo import summary
import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50, resnet34
from torchvision.models.convnext import convnext_tiny
from lightstream.modules.imagenet_template import ImageNetClassifier
from torchmetrics import MetricCollection


# input 320x320x3 on float32, torchinfo
# resnet 34     : Forward/backward pass size (MB): 4286.28
# resnet 50     : Forward/backward pass size (MB): 5806.62
# convnext  tiny: Forward/backward pass size (MB): 4286.28
# inception atto: Forward/backward pass size (MB): 1194.36
# inception tiny: Forward/backward pass size (MB): 3907.07

def _set_layer_scale(model, val=1.0):
    for x in model.modules():
        if hasattr(x, "gamma"):
            x.gamma.data.fill_(val)



def split_net(net, num_classes=1000):
    """Split resnet architectures into backbone and fc modules

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

    stream_net = torch.nn.Sequential(net.stem, net.stages)

    head = net.head

    return stream_net, head


class StreamingInceptionNext(ImageNetClassifier):

    model_choices = {"inceptionnext-atto": inceptionnext_atto, "inceptionnext-tiny": inceptionnext_tiny}

    def __init__(
        self,
        model_name: str,
        tile_size: int,
        loss_fn: torch.nn.functional,
        train_streaming_layers: bool = True,
        metrics: MetricCollection | None = None,
        **kwargs
    ):
        assert model_name in list(StreamingInceptionNext.model_choices.keys())
        network = StreamingInceptionNext.model_choices[model_name](pretrained=True)
        stream_network, head = split_net(network, num_classes=kwargs.pop("num_classes", 1000))

        self._get_streaming_options(**kwargs)
        print("metrics", metrics)
        super().__init__(
            stream_network,
            head,
            tile_size,
            loss_fn,
            train_streaming_layers=train_streaming_layers,
            metrics=metrics,
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
            "before_streaming_init_callbacks": [_set_layer_scale],
        }
        self.streaming_options = {**streaming_options, **kwargs}


if __name__ == "__main__":

    model_default = inceptionnext_atto(pretrained=True)
    batch_size = 16
    summary(model_default, input_size=(batch_size, 3, 320, 320), depth=6)
    #print("hi")

    inputs = torch.randn(1, 3, 320, 320).cuda()

    z_non = model_default(inputs)
    model = StreamingInceptionNext("inceptionnext-atto", 3520, nn.MSELoss)
    model = model.cuda()
    z = model(inputs)



