import torch

from modules.base import BaseModel

from models.resnet.resnet import split_resnet
from models.streamingclam.clam import CLAM_MB, CLAM_SB

from torchvision.models import resnet18, resnet34, resnet50
from torchmetrics.classification import Accuracy, AUROC


# Streamingclam works with resnets, can be extended to other encoders if needed
class CLAMConfig:
    def __init__(
        self,
        encoder: str,
        branch: str,
        n_classes: int = 2,
        gate: bool = True,
        use_dropout: bool = False,
        k_sample: int = 8,
        instance_loss_fn: torch.nn = torch.nn.CrossEntropyLoss,
        subtyping=False,
        *args,
        **kwargs,
    ):
        self.branch = branch
        self.encoder = encoder
        self.n_classes = n_classes
        self.size = self.configure_size()

        self.gate = gate
        self.use_dropout = use_dropout
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping

    def configure_size(self):
        if self.encoder == "resnet50":
            return [2048, 512, 256]
        elif self.encoder in ("resnet18", "resnet34"):
            return [512, 512, 256]

    def configure_model(self):
        # size args original: self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        if self.branch == "sb":
            print("Loading CLAM with single branch \n")
            return CLAM_SB(
                gate=self.gate,
                size=self.size,
                dropout=self.use_dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=self.instance_loss_fn(),
                subtyping=self.subtyping,
            )
        elif self.branch == "mb":
            print("Loading CLAM with multiple branches \n")

            return CLAM_MB(
                gate=self.gate,
                size=self.size,
                dropout=self.use_dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=self.instance_loss_fn(),
                subtyping=self.subtyping,
            )
        else:
            raise NotImplementedError(
                f"branch must be specified as single-branch " f"'sb' or multi-branch 'mb', not {self.branch}"
            )


class StreamingCLAM(BaseModel):
    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

    def __init__(
        self,
        encoder: str,
        tile_size: int,
        loss_fn: torch.nn.functional,
        branch: str,
        n_classes: int,
        max_pool_kernel: int = 0,
        stream_max_pool_kernel: bool = False,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
        *args,
        **kwargs,
    ):
        self.stream_maxpool_kernel = stream_max_pool_kernel
        self.max_pool_kernel = max_pool_kernel
        self.instance_eval = instance_eval
        self.return_features = return_features
        self.attention_only = attention_only

        if self.max_pool_kernel < 0:
            raise ValueError(f"max_pool_kernel must be non-negative, found {max_pool_kernel}")
        if self.stream_maxpool_kernel and self.max_pool_kernel == 0:
            raise ValueError(f"stream_max_pool_kernel cannot be True when max_pool_kernel=0")

        assert encoder in list(StreamingCLAM.model_choices.keys())

        # Define the streaming network and head
        network = StreamingCLAM.model_choices[encoder](weights="IMAGENET1K_V1")
        stream_net, _ = split_resnet(network)
        head = CLAMConfig(encoder=encoder, branch=branch, n_classes=n_classes, **kwargs).configure_model()

        # At the end of the ResNet model, reduce the spatial dimensions with additional max pool
        self.ds_blocks = None
        if self.max_pool_kernel > 0:
            if self.stream_maxpool_kernel:
                stream_net = self.add_maxpool_layers(stream_net)
                super().__init__(stream_net, head, tile_size, loss_fn, *args, **kwargs)
            else:
                ds_blocks, head = self.add_maxpool_layers(head)
                super().__init__(stream_net, head, tile_size, loss_fn, *args, **kwargs)
                self.ds_blocks = ds_blocks

        # Define metrics
        self.train_acc = Accuracy(task="binary", num_classes=n_classes)
        self.train_auc = AUROC(task="binary", num_classes=n_classes)

        self.val_acc = Accuracy(task="binary", num_classes=n_classes)
        self.val_auc = AUROC(task="binary", num_classes=n_classes)

        self.test_acc = Accuracy(task="binary", num_classes=n_classes)
        self.test_auc = AUROC(task="binary", num_classes=n_classes)

    def add_maxpool_layers(self, network):
        ds_blocks = torch.nn.Sequential(torch.nn.MaxPool2d((self.max_pool_kernel, self.max_pool_kernel)))

        if self.stream_maxpool_kernel:
            return torch.nn.Sequential(network, ds_blocks)
        else:
            return ds_blocks, network

    def forward_head(
        self, fmap, mask=None, instance_eval=False, label=None, return_features=False, attention_only=False
    ):
        batch_size, num_features, h, w = fmap.shape

        if self.ds_blocks is not None:
            fmap = self.ds_blocks(fmap)

        # Mask background, can heavily reduce inputs to clam network
        if mask is not None:
            fmap = torch.masked_select(fmap, mask)
            del mask

        # Put everything back together into an array [channels, #unmasked_pixels]
        # Change dimensions from [batch_size, C, H, W] to [batch_size, C, H * W]
        fmap = torch.reshape(fmap, (num_features, -1)).transpose(0, 1)

        if self.attention_only:
            return self.head(fmap, label=None, instance_eval=False, attention_only=self.attention_only)

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.head(
            fmap,
            label=label,
            instance_eval=instance_eval,
            return_features=return_features,
            attention_only=attention_only,
        )

        return logits, Y_prob, Y_hat, A_raw, instance_dict

    def forward(self, x, mask=None):
        if len(x) == 2:
            image, mask = x
        else:
            image = x

        fmap = self.forward_streaming(image)
        out = self.forward_head(
            fmap, mask=mask, return_features=self.return_features, attention_only=self.attention_only
        )
        return out

    def training_step(self, batch, batch_idx: int, *args, **kwargs):
        if len(batch) == 3:
            image, mask, label = batch
        else:
            image, label = batch
            mask = None

        self.image = image
        self.str_output = self.forward_streaming(image)
        self.str_output.requires_grad = self.training

        # Can only be changed when streaming is enabled, otherwise not a leaf variable
        # if self.use_streaming:
        #    self.str_output.requires_grad = True

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.forward_head(
            self.str_output,
            mask=mask,
            instance_eval=self.instance_eval,
            label=label if self.instance_eval else None,
            return_features=self.return_features,
            attention_only=self.attention_only,
        )

        loss = self.loss_fn(logits, label)

        self.train_acc(torch.argmax(logits, dim=1).detach(), label)
        self.train_auc(torch.sigmoid(logits)[:, 1].detach(), label)

        self.log_dict(
            {
                "train_acc": self.train_acc.compute(),
                "train_auc": self.train_auc.compute(),
                "entropy loss": loss.detach(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, label = self._shared_eval_step(batch, batch_idx)
        print(y_hat, label)
        self.val_acc(torch.argmax(y_hat, dim=1), label)
        self.val_auc(torch.sigmoid(y_hat)[:, 1], label)

        # Should update and clear automatically, as per
        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        # https: // lightning.ai / docs / pytorch / stable / extensions / logging.html
        metrics = {"val_acc": self.val_acc, "val_auc": self.val_auc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, y_hat, label = self._shared_eval_step(batch, batch_idx)

        self.test_acc(torch.argmax(y_hat, dim=1))
        self.test_auc(torch.sigmoid(y_hat)[:, 1])

        metrics = {"test_acc": self.test_acc, "test_auc": self.test_auc, "test_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        if len(batch) == 3:
            image, mask, label = batch
        else:
            image, label = batch
            mask = None

        y_hat = self.forward(image)[0].detach()
        loss = self.loss_fn(y_hat, label).detach()

        return loss, y_hat, label.detach()


if __name__ == "__main__":
    model = StreamingCLAM(
        "resnet18",
        tile_size=1600,
        loss_fn=torch.nn.functional.cross_entropy,
        branch="sb",
        n_classes=4,
        max_pool_kernel=8,
    )
