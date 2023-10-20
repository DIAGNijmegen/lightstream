import pyvips
import math
import torch

import pandas as pd
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from lightstream import transforms as T


class StreamingClassificationDataset(Dataset):
    def __init__(
            self,
            img_dir,
            csv_file,
            tile_size,
            img_size,
            transform,
            mask_dir=None,
            mask_suffix="_tissue",
            variable_input_shapes=False,
            tile_delta=None,
            filetype=".tif",
            read_level=0,
            *args,
            **kwargs,
    ):
        self.img_dir = Path(img_dir)
        self.filetype = filetype
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_suffix = mask_suffix

        self.read_level = read_level
        self.tile_size = tile_size
        self.tile_delta = tile_delta
        self.img_size = img_size

        self.variable_input_shapes = variable_input_shapes
        self.transform = transform

        self.classification_frame = pd.read_csv(csv_file)

        # Will be populated in check_csv function
        self.data_paths = {"images": [], "masks": []}

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Directory {self.img_dir} not found or doesn't exist")
        self.check_csv()

    def check_csv(self):
        """Check if entries in csv file exist"""
        included = {"images": [], "masks": []} if self.mask_dir else {"images":[]}
        for i in range(len(self)):
            files = self.get_img_path(i)

            # Files can be just images, but also image, mask
            for file in files:
                if not file.exists():
                    print(
                        f"WARNING {file} not found, excluded both image and mask (if present)!"
                    )
                    continue

            included["images"].append(files[0])

            if self.mask_dir:
                included["masks"].append(files[1])

        self.data_paths = included

    def get_img_path(self, idx):
        img_fname = self.classification_frame.iloc[idx, 0]
        img_path = self.img_dir / Path(img_fname).with_suffix(self.filetype)

        if self.mask_dir:
            mask_path = self.mask_dir / Path(img_fname + self.mask_suffix).with_suffix(
                self.filetype
            )
            return img_path, mask_path

        return [img_path]

    def __getitem__(self, idx):
        img_fname = str(self.data_paths["images"][idx])

        image = pyvips.Image.new_from_file(img_fname, page=self.read_level)
        sample = {"image": image}

        if self.mask_dir:
            mask_fname = str(self.data_paths["masks"][idx])
            mask = pyvips.Image.new_from_file(mask_fname)
            ratio = image.width / mask.width
            print("image width and mask width", image.width, mask.width)
            print("ratio:", ratio, "1/ratio", 1/ratio)
            print(mask.format)
            sample["mask"] = mask.resize(ratio, kernel="nearest")  # Resize mask to img size

        if self.transform:
            print("applying transforms")
            sample = self.transform(**sample)

        # Output of transforms are uint8 images in the range [0,255]
        normalize = T.Compose([
            T.HorizontalFlip(),
            T.ToDtype("float", scale=True),
            T.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        ])

        to_tensor = T.Compose(
            [T.ToTensor(transpose_mask=True)],
            is_check_shapes=False)

        sample = normalize(**sample)

        if self.variable_input_shapes:
            sample = self.pad_to_tile_delta(**sample)
        else:
            sample = self.pad_or_crop_image(**sample)

        # Masks don't need to be really large for tissues, so scale them back
        if self.mask_dir:
            sample["mask"] = sample["mask"].resize(1 / ratio, kernel="nearest")

        sample = to_tensor(**sample)

        if self.mask_dir:
            return sample['image'], sample['mask']

        return sample["image"]

    def __len__(self):
        return len(self.classification_frame)

    def pad_or_crop_image(self, **kwargs):
        """
        TODO: Add this into the albumenations workflow
        """
        image = kwargs.get("image")
        mask = kwargs.get("mask")

        image = image.gravity(
            "centre", self.img_size, self.img_size, background=[255, 255, 255]
        )
        # size of mask should match the size of the image after the encoder + extra downsampling
        if mask:
            mask = mask.gravity(
                "centre", self.img_size, self.img_size, background=[0, 0, 0]
            )

        return image, mask

    def pad_to_tile_delta(self, **kwargs):
        """
        TODO: Add this into the albumenations workflow
        """
        image = kwargs.get("image", None)
        mask = kwargs.get("mask", None)

        if image.width <= self.tile_size:
            w = self.tile_size
        else:
            w = math.ceil(image.width / self.tile_delta) * self.tile_delta
        if image.height <= self.tile_size:
            h = self.tile_size
        else:
            h = math.ceil(image.height / self.tile_delta) * self.tile_delta

        image = image.gravity("centre", w, h, background=[255, 255, 255])

        sample = {"image": image}
        if mask:
            mask = mask.gravity("centre", w, h, background=[0, 0, 0])
            sample["mask"] = mask

        return sample


if __name__ == "__main__":
    root = Path('/data/pathology/projects/pathology-bigpicture-streamingclam')
    data_path = root / Path('data/breast/camelyon_packed_0.25mpp_tif/images')
    mask_path = root / Path('data/breast/camelyon_packed_0.25mpp_tif/images_tissue_masks')
    csv_file = root / Path('streaming_experiments/camelyon/data_splits/train_0.csv')

    dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(csv_file),
        tile_size=1600,
        img_size=3200,
        transform=[],
        mask_dir=mask_path,
        mask_suffix="_tissue",
        variable_input_shapes=True,
        tile_delta=680,
        filetype=".tif",
        read_level=1,
    )

    for x in dataset:
        img, mask = x
        print(img.shape)
