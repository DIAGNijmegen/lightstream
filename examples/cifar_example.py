import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from models.resnet.resnet import StreamingResNet
from torchmetrics import MetricCollection, Accuracy


if __name__ == "__main__":

    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, num_workers=13)

    # Log accuracy during training

    metrics = MetricCollection([
        Accuracy(task="multiclass", num_classes=10)
    ])
    print(metrics)
    model = StreamingResNet(
        "resnet34",
        1920,
        torch.nn.functional.cross_entropy,
        statistics_on_cpu=True,
        verbose=True,
        num_classes=10,
        metrics=metrics
    )

    # train model
    trainer = pl.Trainer(accelerator="gpu", strategy='auto')
    print("the trainer strategy is", trainer.strategy)
    trainer.fit(model=model, train_dataloaders=train_loader)
