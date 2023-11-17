# Image processing using pyvips and albumentationsxl
For this tutorial, we will be using the pyvips backend to load and manipulate images. Pyvips was specifically built with 
large images in mind. It builds data pipelines for each image, rather than directly loading it into memory. As a result,
it can keep a low memory footprint during execution, whilst still being fast. 

Secondly, we will be using the albumentationsxl package. This is virtually the same package as albumentations, but using
the pyvips backend. It features a wide range of image augmentations capable of transforming large images specifically.

Within this tutorial we will be using the Imagenette dataset to serve as an example. Notice that these images are small enough to fit in memory.
The example below only serves as a demonstration of how the data augmentation works with a pyvips backend, and should also work with large images.

## An example using Imagenette

### Downloading and extracting ImageNette data

We start by downloading and extracting the ImageNette dataset into the data folder, which is stored in the current working directory.
The data is then extracted using the `extract_all_files` function

```python
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
```

### Define the dataloader
Defining a pytorch dataset with pyvips is straightforward and does not require much tweaking from normal pipelines. In a nutshell, the following changes are made from a normal pipeline:

- Instead of opening image files using PIL, torch, cv2, etc... we are opening the file with a pyvips backend `pyvips.Image.new_from_file()`
- The albumentations backend is replaced with the albumentationsxl package. 



```python
class ImagenetteDataset(Dataset):
    def __init__(self, patch_size=320, validation=False):
        self.folder = Path("data/imagenette2-320/train") if not validation else Path("data/imagenette2-320/val")
        self.classes = [ "n01440764", "n02102040", "n02979186", "n03000684", "n03028079", "n03394916", "n03417042",
                         "n03425413", "n03445777", "n03888257"]

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
```

Creating custom datasets with large images thus requires little changes and remain flexible enough to use with most of the Pytorch native infrastructure outside of image transformations.
Since these are regular Pytorch Dataset objects, it will work with any data, including masks, bounding boxes, or keypoints, as long as they can be converted into Tensors for model training.


## Image processing best practices
The dataset used in the example enough was small enough to load into memory using normal image libraries. This was done to keep the tutorial lightweight and easy to execute.
In practice larger images will be more of a challenge to optimize correctly. Below we provide several tips to include in your custom pipeline to facilitate fast image processing with lower memory footprints.

## load images as uint8
Pyvips images can be loaded or otherwise cast to uint8 ("uchar" in pyvips). This will increase the speed of the computations done by pyvips,
while also preserving memory. 

## Number of transformations and sequence
Image transformations on large images are costly, therefore, make sure not to include any redundant, e.g. mixing `Affine` with `Rotate` in the same pipeline, as Rotate is a subset of Affine.
Also, some transformations are computationally expensive, such as elastic transforms, so try to avoid using this transformation every time if you are experiencing poor GPU utilization due to cpu bottlenecks.

Finally, the `Normalize` transform will cast the image to a `float32` format. It is recommended to always put this transformation into the very end of the augmentation pipeline, since float32 operations are costlier than `uint8`. Failing to do so can introduce bottlenecks in the augmentation pipeline.

## Optional: image normalization on gpu
within the streaming library, it is possible to normalize the image tile-wise on the gpu instead of in the dataloader. This could improve speed, as well as 
lower memory footprints in some scenarios, for example if the image can be stored as an uint8 tensor before converting to e.g. float16. 


## Image memory requirements
We can take a look at the following table on how images grow in size as we increase their size,
as well as changing their dtypes. From this table, we can already conclude that it is better to work with uint8 and
float16 for training as much a possible. 

Table: All values are in gigabyes (GB). Values generated using random numpy arrays

| Image size    | uint8 | float16 | float32 | float64 |
|---------------|-------|---------|---------|---------|
| 8192x8192x3   | 0.2   | 0.4     | 0.8     | 1.6     |
| 16384x16384x3 | 0.8   | 1.6     | 3.2     | 6.4     |
| 32768x32768x3 | 3.2   | 6.4     | 12.8    | 25.6    |
| 65536x65536x3 | 12.8  | 25.6    | 51.2    | 102.4   |
