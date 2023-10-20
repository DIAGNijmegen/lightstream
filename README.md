# lightstream

## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management - [article](https://mathdatasimplified.com/2023/06/12/poetry-a-better-way-to-manage-python-dependencies/)
* [hydra](https://hydra.cc/): Manage configuration files - [article](https://mathdatasimplified.com/2023/05/25/stop-hard-coding-in-a-data-science-project-use-configuration-files-instead/)
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [DVC](https://dvc.org/): Data version control - [article](https://mathdatasimplified.com/2023/02/20/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-2/)
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Set up the environment
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make env 
```

## Install dependencies
To install all dependencies for this project, run:
```bash
poetry install
```

To install a new package, run:
```bash
poetry add <package-name>
```

## Version your data
To track changes to the "data" directory, type:
```bash
dvc add data
```

This command will create the "data.dvc" file, which contains a unique identifier and the location of the data directory in the file system.

To keep track of the data associated with a particular version, commit the "data.dvc" file to Git:
```bash
git add data.dvc
git commit -m "add data"
```

To push the data to remote storage, type:
```bash
dvc push 
```

## Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs
```

## LightStream

### Image and tile size
- Image size: Multiple of output stride
- Tile size: Multiple of output stride


## Data augmentation pipeline

### Albumentations-like pipeline
- Small differences among pipelines

### Recommended image transformations

#### variable input shapes
Recommend to use variable input shapes when many images are relatively small

- Start with uint8 (uchar) images
- for very large images, use a randomcrop at the start of the augmentation pipeline
- At the end of the augmentation pipeline, some padding or cropping is required to get the correct output in streaming (see above)
  - Strategy 1: Pad the image to be within a multiple of the network output stride
  - Strategy 2: Pad the image to be within a multiple of tile stride, called tile delta. 
  The tile delta is the stride with which the tile moves over the input image during streaming. It is calculated during
  Streaming and its value can be retrieved from the StreamingModule class under ``` _configure_tile_delta``` and be passed on into a data loader
- Convert the image to float32
- Normalize the image

#### Static image sizes
Depending on the dataset, images can be cropped or padded to fixed image sizes at the start of end of the pipeline


- Make sure the images are cropped/padded to an image size that is in a multiple of the network output stride
- If the images are bigger than the chosen image size, crop them at the beginning, if they are smaller, try to avoid padding until the end of the augmentation pipeline
- Near the end of the pipeline, apply a crop_or_pad that crops or pads the image to the chosen image size. Some augmentations like random rotations can change the image size, so it is best to apply this at the end of the augmentation pipeline before casting to floats and normalizing
- Convert the image to float32
- Normalize the image
