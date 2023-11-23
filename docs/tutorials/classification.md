# Basic image classification

This tutorial briefly introduces the `model` repository to easily prototype streaming-capable models right off the bat.
The workflow aims to follow the core design principles of the `lightning` framework, and will not deviate much from it.


## Training a ResNet architecture using streaming
For this example, we will use a ResNet-18 model architecture and train it on the Cifar-10 dataset.
Although this dataset is small enough to train without streaming, we'll use it as a proof of concept.
We assume that the reader is familiar with the regular workflow of a pytorch-lightning model. If this is not the case,
please consult the [lightning](https://lightning.ai/docs/pytorch/stable/) documentation for further information.


### Importing the relevant packages
```python
import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from models.resnet.resnet import StreamingResNet
```

We start by importing the relevant packages. The model repository inside the lightstream package comes with a streaming-capable
version of the ResNet architectures.

### Defining the dataloader and model

The code below defines the datasets and dataloaders, as well as the model definition. The model is trained using the regular
lightning workflow. The `StreamingResNet` class requires you to fill in the following arguments:   

 * `model_name`: A string defining the specific ResNet architecture, in this case ResNet-18
 * `tile_size`: 1600x1660: Streaming processes large images sequentially in **tiles** (or patches), which are stored in a later layer of the model, and then reconstructed into a whole feature map. Higher tile sizes will typically require more VRAM, but will speed up computations.
 * `loss_fn` : `torch.nn.functional.cross_entropy`. The loss function for the network. 
 * `num_classes`: 10. The number of classes to predict. The default is 1000 classes (ImageNet) default. If a different number is specified, then the `fc` layer of the ResNet model is re-initialized with random weight and `num_classes` output neurons.



```python
# Define the dataset, and transform to tensor.
# Since the images in the dataset are small, we don't need pyvips
dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, num_workers=3)

#
model = StreamingResNet(
    "resnet18",  # model architecture arg
    1600,        # the tile size
    torch.nn.functional.cross_entropy, # the loss function, a pytorch object
    num_classes=10
)

# train model
trainer = pl.Trainer(accelerator="gpu")
trainer.fit(model=model, train_dataloaders=train_loader)
```
