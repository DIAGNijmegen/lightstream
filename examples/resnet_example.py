import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from models.resnet.resnet import StreamingResNet
from models.convnext.convnext import StreamingConvnext

dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, num_workers=3)


autoencoder = StreamingResNet(
    "resnet34",
    1920,
    torch.nn.functional.cross_entropy,
    statistics_on_cpu=True,
    verbose=True,
    num_classes=10,
)

"""
autoencoder = StreamingConvnext(
    "convnext_tiny",
    4800,
    torch.nn.functional.cross_entropy,
    statistics_on_cpu=True,
    verbose=True,
    use_streaming=True,
    num_classes=10,
    use_stochastic_depth=False,
)
"""
# train model
trainer = pl.Trainer(accelerator="gpu", strategy='auto')
print("the trainer strategy is", trainer.strategy)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
