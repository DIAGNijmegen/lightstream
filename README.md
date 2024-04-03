# Lightstream

Lightstream is a Pytorch library to train CNN's with large input images. Parsing large inputs is achieved through a combination of 
gradient checkpointing and tiling the input image. For a full overview of the streaming algorithm, please read the article:

[1] H. Pinckaers, B. van Ginneken and G. Litjens, "Streaming convolutional neural networks for end-to-end learning with multi-megapixel images," in IEEE Transactions on Pattern Analysis and Machine Intelligence, [doi: 10.1109/TPAMI.2020.3019563](https://ieeexplore.ieee.org/abstract/document/9178453)


![](docs/images/ddh_08_06_2022.gif)

## Installation
The lightstream repository can be installed using pip, or you can alternatively clone the git repository and build the wheel.

```python
pip install lightstream
```

We also recommend to install the albumentationsxl package, which is an albumentations fork with a pyvips backend to preprocess large images

```python
pip install albumentationsxl
```


### Requirements
The lightstream package requires PyTorch version 2.0 or greater to be installed, along with Pytorch lightning version 2.0.0 or greater. 
- PyTorch 2.0.0 or greater
- Pytorch Lightning 2.0.0 or greater
- Albumentationsxl (recommended)
Furthermore, we recommend a GPU with at least 10 GB of VRAM, and a system with at least 32 GB of RAM.


## Using lightstream with pre-trained networks
lightstream offers several out-of-the-box streaming equivalents of ImageNet classifiers. Currently ResNet and ConvNext architectures are supported

```python
import torch.nn
from lightstream.models.resnet.resnet import StreamingResNet

model = StreamingResNet(model_name="resnet18", tile_size=2880, loss_fn=torch.nn.CrossEntropyLoss(), train_streaming_layers=True)
```

## Documentation
The documentation can be found at https://diagnijmegen.github.io/lightstream/

Alternatively the documentation can be generated locally using 

```
make docs
```
## 