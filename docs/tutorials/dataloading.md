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
