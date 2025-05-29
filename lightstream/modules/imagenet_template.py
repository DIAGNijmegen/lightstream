from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch import LightningModule
from typing import Any
import torch
import torch.distributed as dist
from torch import Tensor

# TODO: Write control flow when lightstream is turned off
# TODO: Add torchmetric collections as parameters (dependency injections)


class ImageNetClassifier(LightningModule):
    def __init__(
        self,
        stream_net: LightningModule,
        head: torch.nn.modules.container.Sequential,
        loss_fn: torch.nn.modules.loss,
        accumulate_grad_batches: int = 1
    ):
        super().__init__()
        self.stream_network = stream_net
        self.head = head
        self.loss_fn = loss_fn
        self.accumulate_grad_batches = accumulate_grad_batches # manual optimization, so do gradient accumulation here

        self.manual_optimization=True


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

        str_output = self.forward_streaming(image)

        # let leaf tensor require grad when training with streaming
        str_output.requires_grad = self.training

        logits = self.forward_head(self.str_output)

        loss = self.loss_fn(logits, target)
        loss = loss / self.accumulate_grad_batches

        self.backward_streaming(loss, image, str_output)

        self.distribute_gradients()

        self.optimizer_step_if_needed(batch_idx)

        output = {}
        output["train_loss"] = loss

        self.log_dict(output, prog_bar=True, on_step=True,  on_epoch=True, sync_dist=True,)
        return loss

    def backward_streaming(self, loss: Tensor, image: Tensor, str_output: Tensor) -> None:
        with self.trainer.strategy.model.no_sync():
            self.manual_backward(loss)
        self.stream_network.backward(image, str_output.grad)


    def distribute_gradients(self):
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(dist.get_world_size())

    def optimizer_step_if_needed(self, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts]

        for opt in opts:
            # accumulate gradients of N batches
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                opt.step()
                opt.zero_grad()

    def validation_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        image, target = batch
        self.image = image

        self.str_output = self.forward_streaming(image)

        # let leaf tensor require grad when training with streaming
        self.str_output.requires_grad = self.training

        logits = self.forward_head(self.str_output)

        loss = self.loss_fn(logits, target)

        output = {}

        output["val_loss"] = loss

        self.log_dict(output, prog_bar=True, on_step=False,  on_epoch=True, sync_dist=True,)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return opt

    def manual_backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        loss.backward()

    def backward(self, loss: Tensor, **kwargs) -> None:
        loss.backward()


