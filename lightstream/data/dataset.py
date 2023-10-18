import pyvips
import pandas as pd

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
        filetype=".tif",
        read_level=0,
        *args,
        **kwargs
    ):
        self.img_dir = Path(img_dir)
        self.filetype = filetype
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix

        self.read_level = read_level
        self.tile_size = tile_size
        self.img_size = img_size

        self.variable_input_shapes = variable_input_shapes
        self.transform = transform

        self.classification_frame = pd.read_csv(csv_file)

        # Will be populated in check_csv function
        self.data_paths = {'images': [], 'masks': []}

        self.check_csv()

    def __getitem__(self, idx):
        img_fname = str(self.data_paths['images'][idx])

        image = pyvips.Image.new_from_file(img_fname, level=self.read_level)
        sample = {'image': image}

        if self.mask_dir:
            mask_fname = str(self.data_paths['images'][idx])
            mask = pyvips.Image.new_from_file((mask_fname))
            sample['mask'] = mask


        if self.transform:
            sample = self.transform(**sample)

        # Padding and cropping
        # Limit size/variable input shapes
        # Normalization
        # To tensor

        return sample

    def __len__(self):
        return len(self.classification_frame)


    def get_img_path(self, idx):
        img_fname = self.classification_frame.iloc[idx, 0]
        img_path = self.img_dir / Path(img_fname).with_suffix(self.filetype)

        if self.mask_dir:
            mask_path = self.mask_dir / Path(img_fname + self.mask_suffix).with_suffix(self.filetype)
            return img_path, mask_path

        return [img_path]

    def check_csv(self):
        """Check if entries in csv file exist"""
        included = {'images': [], 'masks': []}
        for i in range(len(self)):
            files = self.get_img_path(i)

            # Files can be just images, but also image, mask
            for file in files:
                if not file.exists():
                    print("WARNING", file.stem, "not found, excluded both image and mask (if present)!")
                    continue

            included['images'].append(files[0])
            if self.mask_dir:
                included['masks'].append(files[1])

        self.data_paths = included
