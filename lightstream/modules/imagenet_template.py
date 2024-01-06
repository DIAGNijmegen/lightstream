from lightstream.modules.streaming import StreamingModule
from torchmetrics import MetricCollection
from typing import Any
import torch


# TODO: Write control flow when lightstream is turned off
# TODO: Add torchmetric collections as parameters (dependency injections)


class ImageNetClassifier(StreamingModule):
    def __init__(
        self,
        stream_net: torch.nn.modules.container.Sequential,
        head: torch.nn.modules.container.Sequential,
        tile_size: int,
        loss_fn: torch.nn.modules.loss,
        train_streaming_layers=True,
        metrics: MetricCollection | None = None,
        **kwargs
    ):
        super().__init__(stream_net, tile_size, train_streaming_layers, **kwargs)
        self.head = head
        self.loss_fn = loss_fn
        self.train_streaming_layers = train_streaming_layers
        self.params = self.extend_trainable_params()

        self.train_metrics = metrics.clone(prefix='train_') if metrics else None
        self.val_metrics = metrics.clone(prefix='val_') if metrics else None
        self.test_metrics = metrics.clone(prefix='test_') if metrics else None

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

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

        # let leaf tensor require grad when training with streaming
        self.str_output.requires_grad = self.training

        logits = self.forward_head(self.str_output)

        loss = self.loss_fn(logits, target)

        output = {}
        if self.train_metrics:
            output = self.train_metrics(logits, target)

        output["train_loss"] = loss

        self.log_dict(output, prog_bar=True, on_step=True,  on_epoch=True, sync_dist=True,)
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
        # del loss
        # Don't call this>? https://pytorch-lightning.readthedocs.io/en/1.5.10/guides/speed.html#things-to-avoid
        torch.cuda.empty_cache()
        if self.train_streaming_layers:
            self.backward_streaming(self.image, self.str_output.grad)
        del self.str_output
