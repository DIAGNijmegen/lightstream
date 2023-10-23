import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import lightning.pytorch as pl

# My own edits here

from models.streamingclam.streamingclam import StreamingCLAM

transforms = transforms.Compose([transforms.Resize((1600, 1600)), transforms.ToTensor()])
dataset = CIFAR10(os.getcwd(), download=True, transform=transforms)
train_loader = DataLoader(dataset, num_workers=0)

model = StreamingCLAM(
    "resnet18",
    tile_size=1600,
    loss_fn=torch.nn.functional.cross_entropy,
    branch="sb",
    n_classes=2,
    max_pool_kernel=8,
    statistics_on_cpu=True,
    verbose=False,
)


# train model
trainer = pl.Trainer(accelerator="gpu")
trainer.fit(model=model, train_dataloaders=train_loader)
