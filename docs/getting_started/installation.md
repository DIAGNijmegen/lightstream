# Lightstream

Lightstream is a Pytorch-Lightning library for training CNN-based models with large input data using streaming. 
This approach allows you to parse huge (image) inputs through a CNN without running into memory bottlenecks, i.e. getting GPU out of memory (OOM) errors.

The underlying algorithm is based on the `streaming` paper described in [[1]](#1). During training/inferencing, 
a huge input image that would normally cause GPU OOM is split into tiles and processed sequentially until a pre-defined part of the network. 
There, the individual tiles are stitched back together, and the forward/backward is finished normally. Due to gradient 
checkpointing, intermediate features are deleted to save memory, and are re-computed tile-wise during backpropagation (see figure below)
[^1] The exact method 

By doing so, the result is mathematically the same as if the large input was parsed directly through a GPU without memory restrictions.



## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

## References
<a id="1">[1]</a> 
H. Pinckaers, B. van Ginneken and G. Litjens,
"Streaming convolutional neural networks for end-to-end learning with multi-megapixel images,"
in IEEE Transactions on Pattern Analysis and Machine Intelligence, 
[doi: 10.1109/TPAMI.2020.3019563](https://ieeexplore.ieee.org/abstract/document/9178453)

