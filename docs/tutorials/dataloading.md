# Image processing using pyvips and albumentationsxl
For this tutorial, we will be using the pyvips backend to load and manipulate images. Pyvips was specifically built with 
large images in mind. It builds data pipelines for each image, rather than directly loading it into memory. As a result,
it can keep a low memory footprint during execution, whilst still being fast. 

Secondly, we will be using the albumentationsxl package. This is virtually the same package as albumentations, but using
the pyvips backend. It features a wide range of image augmentations capable of transforming large images specifically.

Before we start, we can take a look at the following table on how images grow in size as we increase their size,
as well as changing their dtypes. From this table, we can already conclude that it is better to work with uint8 and
float16 for training as much a possible. 

Table: All values are in gigabyes (GB). Values generated using random numpy arrays

| Image size    | uint8 | float16 | float32 | float64 |
|---------------|-------|---------|---------|---------|
| 8192x8192x3   | 0.2   | 0.4     | 0.8     | 1.6     |
| 16384x16384x3 | 0.8   | 1.6     | 3.2     | 6.4     |
| 32768x32768x3 | 3.2   | 6.4     | 12.8    | 25.6    |
| 65536x65536x3 | 12.8  | 25.6    | 51.2    | 102.4   |


## An example using cifar 10
TODO


## Image processing best practices



## load images as uint8
Pyvips images can be loaded or otherwise cast to uint8 ("uchar" in pyvips). This will increase the speed of the computations done by pyvips,
while also preserving memory. 

## Number of transformations and sequence
Image transformations on large images are costly, therefore, make sure not to include any redundant, e.g. mixing `Affine` with `Rotate` in the same pipeline, as Rotate is a subset of Affine.
Also, some transformations are computationally expensive, such as elastic transforms, so try to avoid using this transformation every time if you are experiencing poor GPU utilization due to cpu bottlenecks.

Finally, the `Normalize` transform will cast the image to a `float32` format. It is recommended to always put this transformation into the very end of the augmentation pipeline, since float32 operations are costlier than `uint8`. Failing to do so can introduce bottlenecks in the augmentation pipeline.

### Best practice ??: Image normalization
Image normalization is usually the step that is performed on a float format in tensor space. The lightstream library allows for tile-wise
image normalization on the gpu. 