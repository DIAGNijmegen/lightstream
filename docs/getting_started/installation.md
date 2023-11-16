# Installation

## Installing lightstream
Installing the lightstream package can be done via pip:
``` py
pip install lightstream
```

Alternative, the project can be cloned using `git` from the [lightstream git repo](https://github.com/DIAGNijmegen/lightstream). 
From there, the package can again be installed using `pip install .` within the directory. Otherwise, you can use the repository as-is, but
you will have to add it to your `PYTHONPATH` if you want to access it from other repositories.

## Installing (Py)vips
Before pyvips can be installed, libvips must be present on the system, as pyvips is a Python wrapper around the libvips library.
Libvips can be installed either using the binaries on the [libvips website](https://www.libvips.org/install.html), or it can be built from source with or without Docker.
After libvips is properly installed, pyvips can be installed via pip:
``` py
pip install pyvips
```

## Installing torch-related libraries
Installing torch and its related libraries (lightning, torchvision) can be done via pip installs. We recommend looking at their respective websites
for further help to install these packages.
