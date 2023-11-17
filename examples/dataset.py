import os
import pyvips
import albumentationsxl as A
from pathlib import Path
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import tarfile


def extract_all_files(tar_file_path, extract_to):
    with tarfile.open(tar_file_path, "r") as tar:
        tar.extractall(extract_to)


def download_and_extract():
    # Download dataset and extract in a data/ directory
    if not os.path.isfile("data/imagenette2-320.tgz"):
        print("Downloading dataset")
        download_url("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", os.getcwd() + "/data")
        print("Extracting dataset")
        extract_all_files(os.getcwd() + "/data/imagenette2-320.tgz", os.getcwd() + "/data")


class ImagenetteDataset(Dataset):
    def __init__(self, patch_size=320, validation=False):
        self.folder = Path("data/imagenette2-320/train") if not validation else Path("data/imagenette2-320/val")
        self.classes = [
            "n01440764",
            "n02102040",
            "n02979186",
            "n03000684",
            "n03028079",
            "n03394916",
            "n03417042",
            "n03425413",
            "n03445777",
            "n03888257",
        ]

        self.images = []
        for cls in self.classes:
            cls_images = list(self.folder.glob(cls + "/*.JPEG"))
            self.images.extend(cls_images)

        self.patch_size = patch_size
        self.validation = validation

        self.transforms = A.Compose(
            [
                A.RandomBrightnessContrast(p=1.0),
                A.Rotate(p=0.8),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        image_fname = self.images[index]
        image = pyvips.Image.new_from_file(image_fname)
        label = image_fname.parent.stem
        label = self.classes.index(label)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    download_and_extract()

    ds = ImagenetteDataset()
    for x in ds:
        print(x)