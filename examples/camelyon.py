import pyvips
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pathlib import Path

import pandas as pd

from lightstream.modules import ImageNetClassifier
from lightstream.models.resnet import StreamingResNet
from lightning.pytorch import Trainer

class CamelyonDataset(Dataset):
    def __init__(self, image_dir, label_csv, mode="train"):
        super().__init__()

        self.image_dir = image_dir
        self.df = pd.read_csv(label_csv)

        if mode == "train":
            self.df = self.df[~self.df["image"].str.startswith("test_")]
        else:
            self.df = self.df[self.df["image"].str.startswith("test_")]

        self.df['label'] = self.df['type'].apply(lambda x: 1 if x == "tumor" else 0)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        label = self.df["label"].iloc[item]
        img_name = self.df["image"].iloc[item]
        img_path = self.image_dir / Path(img_name)

        img = pyvips.Image.new_from_file(str(img_path), page=4)
        img = torch.Tensor(img.numpy()).permute(2,0,1)  # Don't need to normalize, is done within streaming on gpu

        return img, label

def configure_model(encoder="resnet18", tile_size=2560):
    stream_model = StreamingResNet(encoder, tile_size=tile_size)
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1) , nn.Flatten(), nn.Linear(512, 2))
    loss_fn = nn.CrossEntropyLoss()
    return ImageNetClassifier(stream_model, head, loss_fn, accumulate_grad_batches=1)


if __name__ == "__main__":
    encoder = "resnet18"
    tile_size = 2560
    label_csv_path = Path.cwd() / Path("reference.csv")
    image_dir = "/data/temporary/archives/breast/camelyon/camelyon16_packed_0.5mpp/packed"

    model = configure_model(encoder, tile_size)

    # Since this is just an example, just validate on the test set.
    train_dataset = CamelyonDataset(image_dir, label_csv_path, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = CamelyonDataset(image_dir, label_csv_path, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    trainer = Trainer(devices=1, accelerator='gpu')
    trainer.fit(model, train_dataloader, test_dataloader)

