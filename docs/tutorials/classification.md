# Basic image classification

This tutorial briefly introduces the `model` repository to easily prototype streaming-capable models right off the bat.
The workflow aims to follow the core design principles of the `lightning` framework, and will not deviate much from it.


## Training a ResNet architecture using streaming
For this example, we will use a ResNet-18 model architecture and train it on the Camelyon16 dataset.
We assume that the reader is familiar with the regular workflow of a pytorch-lightning model. If this is not the case,
please consult the [lightning](https://lightning.ai/docs/pytorch/stable/) documentation for further information.


### Importing the relevant packages
```python
import torch
import pyvips
import pandas as pd
import albumentationsxl as A

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from lightstream.models.resnet.resnet import StreamingResNet
from sklearn.model_selection import train_test_split
from lightning.pytorch import Trainer
```

We start by importing the relevant packages. The model repository inside the lightstream package comes with a streaming-capable
version of the ResNet architectures. We also recommend installing the [albumentationsxl](https://github.com/stephandooper/albumentationsxl) package.
This package is an albumentations-like augmentation package with a pyvips backend to facilitate loading and transforming large images.

Note that for this example, we will be training on a relatively low resolution, since training on high resolutions take a long time to finish.

### Some general settings

We define the paths to the data directory where the images reside, as well as open the label csv file with the image names and labels. For this example we are solving a binary classification problem: tumor versus no tumor within a slide.
The dataset is split into a training, validation, and test set, and we define the desired transformations using the albumentationsxl package. For this example, we will only use flips, and then cast them to float tensors.

```python
ROOT_DIR = Path("/data/pathology/archives/breast/camelyon/CAMELYON16")
label_df = pd.read_csv(str(ROOT_DIR / Path("evaluation/reference.csv")))
image_dir = ROOT_DIR / Path("images")

label_df["label"] = label_df["class"].apply(lambda x: 0 if x =="negative" else 1)
#%%
test_df = label_df[label_df["image"].str.startswith("test")]
train_df = label_df[label_df["image"].str.startswith("normal") | label_df["image"].str.startswith("tumor")]
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42,stratify=train_df["label"])
#%%
# Normalizing with imagenet statistics is done during streaming in scnn.py, so we don't do that here
image_size=4096
train_transforms  = A.Compose([A.CropOrPad(image_size, image_size), A.Flip(p=0.5), A.ToDtype("float", scale=True), A.ToTensor()])
test_transforms = A.Compose([A.CropOrPad(image_size, image_size), A.ToDtype("float", scale=True), A.ToTensor()])

```



### Defining the dataloader and model

We first define the dataset class for the Camelyon16 dataset. We only need the csv file with image names and the label, as well as the path to the image directory.
The albumentationsxl package also makes it a straightforward process to open and additionally augment images. Finally, we construct the dataloaders with 1 worker. Typically, you would want to set this number higher, but remember that we are working with large images, so setting `num_workers` to a high value can lead to out of memory errors.

```python
class CamelyonDataset(Dataset):
    def __init__(self, image_dir: list, df: pd.DataFrame, transform: A.Compose| None=None):
        self.image_dir = image_dir
        self.df = df
        self.transforms = transform
        self.df["image_path"] = self.df["image"].apply(lambda x: image_dir / Path(x).with_suffix(".tif"))
        
        self.images = self.df["image_path"].tolist()
        self.labels = self.df["label"].tolist()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        try:
            image = pyvips.Image.new_from_file(self.images[item], level=5)[0:3]
        except Exception as e:
            image = pyvips.Image.new_from_file(self.images[item], page=5)[0:3]
            
        label = self.labels[item]
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        return image, label


train_loader = DataLoader(dataset=CamelyonDataset(image_dir=image_dir, df=train_df, transform=train_transforms), num_workers=1)
valid_loader = DataLoader(dataset=CamelyonDataset(image_dir=image_dir, df=val_df, transform=test_transforms), num_workers=1)
test_loader = DataLoader(dataset=CamelyonDataset(image_dir=image_dir, df=test_df, transform=test_transforms), num_workers=1)
```


#### Defining the streaming model
We will now define a ResNet model that is streamable. The code can be found below. We initialize the model with the following values:

 * `model_name`: A string defining the specific ResNet architecture, in this case ResNet-18
 * `tile_size`: 2880x2880: Streaming processes large images sequentially in **tiles** (or patches), which are stored in a later layer of the model, and then reconstructed into a whole feature map. Higher tile sizes will typically require more VRAM, but will speed up computations.
 * `loss_fn` : `torch.nn.functional.cross_entropy`. The loss function for the network. 
 * `num_classes`: 2. The number of classes to predict. The default is 1000 classes (ImageNet) default. If a different number is specified, then the `fc` layer of the ResNet model is re-initialized with random weight and `num_classes` output neurons.
 * `train_streaming_layers`: False. Whether to train the streaming layers of the ResNet18, we set it to false here (see the reason why below)


```python
model = StreamingResNet(model_name="resnet18", tile_size=2880, num_classes=2, train_streaming_layers=False, loss_fn=torch.nn.CrossEntropyLoss())
```
With this function, we defined an ImageNet pre-trained ResNet-18 model where all of the ResNet-18 layers are now streamed instead of trained directly. The final layers of the model, specifically the global pooling layers and fully connected layers, are not streamed/streamable.
Notice that we set `train_streaming_layers=False` here. 

This means that only the parameters of the head classifier will be trainable. We do this for two reasons. First, the classifier head is randomly initialized, and it is recommended to first finetune the randomized parameters in the model, keeping the ImageNet weights frozen, to stabilize training and prevent divergence.
Secondly, training all the parameters in the model will add additional computational time during the backward pass. In this example, we do not care about achieving SOTA, but rather show how to train a streaming model.

When the streaming function is executed, you will usually see an output similar to what is shown below. These are tile statistics that are calculated under the hood and it shows how many pixels are invalid due to padding within the layers.
This is shown for each consecutive convolution and pooling layer within the model. If at any point the sum of the bottom+top, or left+right statistics exceed the tile size for any layer, then the model will not be properly defined, and you will need to **increase** the tile size.


```out
metrics None
No tile cache found, calculating it now

Converting modules to nn.Identity()
Executing pre-streaming initialization callbacks (if any):

Initializing streaming model
Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
 Lost(top:2.0, left:2.0, bottom:1.0, right:1.0)
MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) 
 Lost(top:2.0, left:2.0, bottom:1.0, right:1.0)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:3.0, left:3.0, bottom:2.0, right:2.0)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:4.0, left:4.0, bottom:3.0, right:3.0)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:5.0, left:5.0, bottom:4.0, right:4.0)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:6.0, left:6.0, bottom:5.0, right:5.0)
Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 
 Lost(top:4.0, left:4.0, bottom:3.0, right:3.0)
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:5.0, left:5.0, bottom:4.0, right:4.0)
Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False) 
 Lost(top:3.0, left:3.0, bottom:2.0, right:2.0)
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:6.0, left:6.0, bottom:5.0, right:5.0)
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:7.0, left:7.0, bottom:6.0, right:6.0)
Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 
 Lost(top:4.0, left:4.0, bottom:3.0, right:3.0)
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:5.0, left:5.0, bottom:4.0, right:4.0)
Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False) 
 Lost(top:4.0, left:4.0, bottom:3.0, right:3.0)
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:6.0, left:6.0, bottom:5.0, right:5.0)
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:7.0, left:7.0, bottom:6.0, right:6.0)
Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 
 Lost(top:4.0, left:4.0, bottom:3.0, right:3.0)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:5.0, left:5.0, bottom:4.0, right:4.0)
Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False) 
 Lost(top:4.0, left:4.0, bottom:3.0, right:3.0)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:6.0, left:6.0, bottom:5.0, right:5.0)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:7.0, left:7.0, bottom:6.0, right:6.0)

 Output lost Lost(top:7.0, left:7.0, bottom:6.0, right:6.0)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:7.0, left:7.0, bottom:6.0, right:6.0)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:8.0, left:8.0, bottom:7.0, right:7.0)
Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False) 
 Lost(top:9.0, left:9.0, bottom:8.0, right:8.0)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:9.0, left:9.0, bottom:8.0, right:8.0)
Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 
 Lost(top:10.0, left:10.0, bottom:9.0, right:9.0)
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:22.0, left:22.0, bottom:21.0, right:21.0)
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:23.0, left:23.0, bottom:22.0, right:22.0)
Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False) 
 Lost(top:24.0, left:24.0, bottom:23.0, right:23.0)
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:24.0, left:24.0, bottom:23.0, right:23.0)
Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 
 Lost(top:25.0, left:25.0, bottom:24.0, right:24.0)
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:52.0, left:52.0, bottom:51.0, right:51.0)
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:52.0, left:52.0, bottom:51.0, right:51.0)
Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False) 
 Lost(top:56.0, left:56.0, bottom:55.0, right:55.0)
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:56.0, left:56.0, bottom:55.0, right:55.0)
Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 
 Lost(top:55.0, left:55.0, bottom:54.0, right:54.0)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:112.0, left:112.0, bottom:111.0, right:111.0)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:113.0, left:113.0, bottom:112.0, right:112.0)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:120.0, left:120.0, bottom:119.0, right:119.0)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
 Lost(top:119.0, left:119.0, bottom:118.0, right:118.0)
testing shape gradient fix
MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) 
 Lost(top:120.0, left:120.0, bottom:119.0, right:119.0)
Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
 Lost(top:241.0, left:241.0, bottom:239.0, right:239.0)

 Input gradient lost Lost(top:490.0, left:490.0, bottom:487.0, right:487.0)
Restoring model weights
Executing post-streaming initialization callbacks (if any):

writing streaming cache file to /tmp/pycharm_project_341/notebooks/tile_cache_1_3_2880_2880
WARNING: Streaming network will not be trained

```

## Training the model

Finally, we define a regular pytorch lightning trainer and we can fit the model as usual.

```python
model = StreamingResNet(model_name="resnet18", tile_size=2880, num_classes=2, train_streaming_layers=False, loss_fn=torch.nn.CrossEntropyLoss())
#%%
trainer = Trainer(
        default_root_dir="./",
        accelerator="gpu",
        max_epochs=15,
        devices=1,
        precision="16-mixed",
        strategy="auto",
    )

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
```
