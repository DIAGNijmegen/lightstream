import pyvips
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd


from lightstream.modules import ImageNetClassifier
from lightstream.models.resnet import StreamingResNet

class CamelyonDataset(Dataset):
    def __init__(image_dir, label_csv):
        super().__init__()

def configure_model(encoder="resnet18", tile_size=2560):
    stream_model = StreamingResNet(encoder, tile_size=tile_size)
    head = nn.Sequential(nn.AdaptiveAvgPool2d(512), nn.Flatten(), nn.Linear(512, 2))
    loss_fn = nn.CrossEntropyLoss()
    return ImageNetClassifier(stream_model, head, loss_fn, accumulate_grad_batches=1)



if __name__ == "__main__":
    encoder = "resnet18"
    tile_size = 2560

    model = configure_model(encoder, tile_size)
