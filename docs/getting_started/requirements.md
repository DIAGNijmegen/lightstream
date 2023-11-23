# Requirements

## Software Requirements
The lightstream package is supported on Linux distributions. Although it should work on Windows, it is untested and therefore not recommended.

The following packages will need to be installed for lightstream to work. Note that albumentationsXL is not strictly required,
but features an albumentations-like data augmentations pipeline for processing large images.


- [Pytorch](https://pytorch.org/) (2.0.0) or higher
- [Pytorch-lightning](https://lightning.ai/) (2.0.0) or higher
- [torchvision](https://pytorch.org/vision/stable/index.html) (0.15 or higher)
- [Pyvips](https://github.com/libvips/pyvips)
  - Requires [libvips](https://github.com/libvips/libvips) to be installed and configured.
- [albumentationsXL](https://github.com/stephandooper/albumentationsxl) (optional, but recommended)


## Hardware requirements
* **GPU**: Considering lightstream is computationally heavy, it is highly recommend to use a GPU with at least 10GB VRAM instead of CPU backends.

* **RAM**: Additionally, large amounts of RAM are highly recommended (at least 32 GB). Although pyvips will keep memory footprints low during execution, they will
have to be stored in RAM before sending it over to the GPU. Since we are dealing with large images, this can become quite costly.
To give an idea of the RAM cost, we have included a reference table of the approximate memory footprint of several image sizes with varying dtypes.
Given the values in [table 1](#_table-1), it is recommended to work with np.uint8 as much as possible, and only use float32 sparingly. 
Float64 images are usually unnecessary and should be avoided entirely.
Table: All values are in gigabyes (GB)

| Image size    | uint8 | float16 | float32 | float64 |
|---------------|-------|---------|---------|---------|
| 8192x8192x3   | 0.2   | 0.4     | 0.8     | 1.6     |
| 16384x16384x3 | 0.8   | 1.6     | 3.2     | 6.4     |
| 32768x32768x3 | 3.2   | 6.4     | 12.8    | 25.6    |
| 65536x65536x3 | 12.8  | 25.6    | 51.2    | 102.4   |
