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
        accumulate_grad_batches=32,
        train_streaming_layers=True,
        use_streaming=True,
        *args,
        **kwargs
    ):
        super().__init__(
            stream_net,
            tile_size,
            use_streaming,
            train_streaming_layers,
            *args,
            **kwargs
        )
        self.head = head
        self.loss_fn = loss_fn
        self.train_streaming_layers = train_streaming_layers
        self.params = self.extend_trainable_params()
        self.accumulate_batches = accumulate_grad_batches

    def on_train_epoch_start(self) -> None:
        print("on train epoch start hook")
        print("Setting batchnorm layers to eval")
        self.freeze_streaming_normalization_layers()

    def forward_head(self, x):
        return self.head(x)

    def forward(self, x):
        fmap = self.forward_streaming(x)
        out = self.forward_head(fmap)
        return out

    def training_step(
        self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[Any, Any, Any]:
        image, target = batch
        opt = self.optimizers()

        fmap = self.forward_streaming(image)
        # Can only be changed when streaming is enabled, otherwise not a lead variable
        if self.use_streaming:
            fmap.requires_grad = True

        output_head = self.forward_head(fmap)
        loss = self.loss_fn(output_head, target) / self.accumulate_batches

        self.manual_backward(loss, batch, fmap, output_head)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.accumulate_batches == 0:
            opt.step()
            opt.zero_grad()

        self.log_dict({"entropy loss": loss}, prog_bar=True)

    def manual_backward(
        self,
        loss: torch.Tensor,
        batch: Any,
        fmap: Any,
        out: Any,
        *args: Any,
        **kwargs: Any
    ):
        """

        Parameters
        ----------
        loss : torch.Tensor, the loss of the model
        batch : The NHWC input image to the model
        out: The (fmap, out) output tuple from the training step. Needed for fmap.grad
        args
        kwargs

        Returns
        -------

        """

        # if use_streaming is False, backward through the stream_network is controlled by loss.backward()
        loss.backward()
        input_image, _ = batch
        if self.train_streaming_layers and self.use_streaming:
            self.backward_streaming(input_image, fmap.grad)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.params, lr=1e-3)
        return opt

    def extend_trainable_params(self):
        return self.params + list(self.head.parameters())
