from lightning.pytorch.utilities.types import OptimizerLRScheduler

from lightstream.modules.streaming import StreamingModule
from typing import Any, Tuple
import torch


# TODO: Write control flow when lightstream is turned off


class BaseModel(StreamingModule):
    def __init__(
        self,
        stream_net: torch.nn.modules.container.Sequential,
        head: torch.nn.modules.container.Sequential,
        tile_size: int,
        loss_fn: torch.nn.modules.loss,
        train_streaming_layers=True,
        use_streaming=True,
        *args,
        **kwargs
    ):
        super().__init__(stream_net, tile_size, use_streaming, train_streaming_layers, *args, **kwargs)
        self.head = head
        self.loss_fn = loss_fn
        self.train_streaming_layers = train_streaming_layers
        self.params = self.extend_trainable_params()

    def on_train_epoch_start(self) -> None:
        self.freeze_streaming_normalization_layers()

    def forward_head(self, x):
        return self.head(x)

    def forward(self, x):
        fmap = self.forward_streaming(x)
        out = self.forward_head(fmap)
        return out

    def training_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        image, target = batch
        self.image = image

        self.str_output = self.forward_streaming(image)
        self.str_output.requires_grad = self.training

        out = self.forward_head(self.str_output[0])
        loss = self.loss_fn(out, target)

        self.log_dict({"entropy loss": loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.params, lr=1e-3)
        return opt

    def extend_trainable_params(self):
        if self.params:
            return self.params + list(self.head.parameters())
        return list(self.head.parameters())

    def backward(self, loss):
        loss.backward()
        del loss

        # Don't call this>? https://pytorch-lightning.readthedocs.io/en/1.5.10/guides/speed.html#things-to-avoid
        torch.cuda.empty_cache()
        if self.train_streaming_layers and self.use_streaming:
            with torch.set_grad_enabled(True):
                self.backward_streaming(self.image[None], self.str_output.grad)
        del self.str_output
