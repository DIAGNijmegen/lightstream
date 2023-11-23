# Lightstream

Lightstream is a Pytorch-Lightning library for training CNN-based models with large input data using streaming. 
This approach allows you to parse huge (image) inputs through a CNN without running into memory bottlenecks, i.e. getting GPU out of memory (OOM) errors.

The underlying algorithm is based on the `streaming` paper described in [[1]](#1). During training/inferencing, 
a huge input image that would normally cause GPU OOM is split into tiles and processed sequentially until a pre-defined part of the network. 
There, the individual tiles are stitched back together, and the forward/backward is finished normally. Due to gradient 
checkpointing, intermediate features are deleted to save memory, and are re-computed tile-wise during backpropagation (see figure below).

By doing so, the result is mathematically the same as if the large input was parsed directly through a GPU without memory restrictions.


![Alt Text](images/ddh_08_06_2022.gif)


## Implemented in Pytorch-Lightning
The Lightstream package is simple to test and extend as it works with native Pytorch, and also works with Lightning to minimize boilerplate code.
Most convolutional neural networks can be easily converted into a streaming equivalent using a simple wrapper in native Pytorch:

```python
# Resnet18 turned into a streaming-capable network
from torchvision.models import resnet18
import torch.nn as nn

stream_layers = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
    )


sCNN = StreamingCNN(stream_layers, tile_shape=(1, 3, 600, 600))
str_output = sCNN(image)

final_output = final_layers(str_output)
loss = criterion(final_output, labels)

loss.backward()
sCNN.backward(image, str_output.grad)

```

!!! warning

    Not all layers are supported during streaming. To see the caveats, please consult the how-to pages.



## Limitations to streaming
The streaming algorithm exploits the fact that convolutions are locally defined. This means that the entire
input image does not have to be parsed through the network at once, but can be reconstructed piece-wise. There
are many layers that do not possess this property, such as batch normalization layers, where the mean and standard deviations
computations require the entire image at once.


### Layers that can be used without issue

- Convolutional layers: can be used without issue, since they are locally defined
- Pooling layers such as average, max, GEM pooling, as long as they are locally defined, e.g. a 2x2 kernel. Global pooling will not work.
- Any other layer that is defined locally and not dependent on seeing the entire image at once. 

### Layers that are restricted
- All normalization layers: e.g. batch normalization. Most normalization layers require image-level statistics such as means and standard deviations to be computed. As streaming works tile-wise, they will not yield the correct results. Therefore, all normalization layers must be set to ```eval()``` during training.
- **Dense layers** can only be used to model 1x1 convolutions, i.e. a fully connected layer that works over the channels of an input, rather than the spatial dimensions.





    
## References
<a id="1">[1]</a> 
H. Pinckaers, B. van Ginneken and G. Litjens,
"Streaming convolutional neural networks for end-to-end learning with multi-megapixel images,"
in IEEE Transactions on Pattern Analysis and Machine Intelligence, 
[doi: 10.1109/TPAMI.2020.3019563](https://ieeexplore.ieee.org/abstract/document/9178453)

