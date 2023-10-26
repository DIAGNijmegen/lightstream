import torch

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from pathlib import Path

from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torch.utils.data import DataLoader

# My own edits here
from models.streamingclam.streamingclam import StreamingCLAM
from data.dataset import StreamingClassificationDataset


def weighted_sampler(dataset):
    labels = np.array([int(label) for label in dataset.labels])

    # more generalized approach, should result in the same distribution
    # calculate inverse class frequency, then squash to [0,1] by dividing by max value
    _, class_counts = np.unique(labels, return_counts=True)
    inv_freq = len(labels) / class_counts
    norm_weights = inv_freq / np.max(inv_freq)

    # create weight array and replace labels by their weights
    weights = np.array(labels, dtype=np.float32)
    for i, weight in enumerate(norm_weights):
        weights[labels == i] = weight

    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


if __name__ == "__main__":
    root = Path("/opt/ml")
    data_path = root / Path("/input/data/images")
    mask_path = root / Path("/input/data/tissue_masks")
    train_csv = root / Path("/input/data/train.csv")
    val_csv = root / Path("/input/data/val.csv")
    test_csv = root / Path("/input/data/test.csv")

    model = StreamingCLAM(
        "resnet18",
        tile_size=9984,
        loss_fn=torch.nn.functional.cross_entropy,
        branch="sb",
        n_classes=2,
        max_pool_kernel=0,
        statistics_on_cpu=True,
        verbose=False,
        train_streaming_layers=False,
    )

    tile_delta = model._configure_tile_delta()
    network_output_stride = max(
        model.stream_network.output_stride[1] * model.max_pool_kernel, model.stream_network.output_stride[1]
    )

    train_dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(train_csv),
        tile_size=9984,
        img_size=32768,
        transform=[],
        mask_dir=mask_path,
        mask_suffix="",
        variable_input_shapes=True,
        tile_delta=tile_delta,
        network_output_stride=network_output_stride,
        filetype=".tif",
        read_level=1,
    )

    val_dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(val_csv),
        tile_size=9984,
        img_size=32768,
        transform=[],
        mask_dir=mask_path,
        mask_suffix="",
        variable_input_shapes=True,
        tile_delta=tile_delta,
        network_output_stride=network_output_stride,
        filetype=".tif",
        read_level=1,
    )

    test_dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(test_csv),
        tile_size=9984,
        img_size=32768,
        transform=[],
        mask_dir=mask_path,
        mask_suffix="",
        variable_input_shapes=True,
        tile_delta=tile_delta,
        network_output_stride=network_output_stride,
        filetype=".tif",
        read_level=1,
    )

    sampler = weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, num_workers=3, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, num_workers=3, shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="streamingclam-derma-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        save_last=False,
        mode="min",
        every_n_epochs=1,
        verbose=True,
    )

    # train model
    trainer = pl.Trainer(
        default_root_dir="/opt/ml/checkpoints",
        accelerator="gpu",
        max_epochs=2,
        check_val_every_n_epoch=1,
        devices=8,
        strategy="ddp",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
