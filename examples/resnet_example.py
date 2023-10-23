import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import lightning.pytorch as pl

# My own edits here

from models.resnet.resnet import StreamingResNet




dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, num_workers=7)

autoencoder= StreamingResNet('resnet18', 1600, torch.nn.functional.cross_entropy, statistics_on_cpu=True, verbose=False)


# train model
trainer = pl.Trainer(accelerator="gpu")
trainer.fit(model=autoencoder, train_dataloaders=train_loader)