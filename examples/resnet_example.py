import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import lightning.pytorch as pl

# My own edits here

from models.resnet.resnet import StreamingResNet

dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, num_workers=3)

autoencoder = StreamingResNet(
    "resnet34",
    1920,
    torch.nn.functional.cross_entropy,
    statistics_on_cpu=True,
    verbose=True,
    use_streaming=True,
    num_classes=10,
)

# train model
trainer = pl.Trainer(accelerator="gpu")
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
