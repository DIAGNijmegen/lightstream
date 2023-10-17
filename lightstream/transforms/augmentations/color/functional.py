"""

Functional representations of the augmentations. Most will function as functional pyvips aliases.

"""

import pyvips
import numpy as np

__all__ = ["separate_stains", "combine_stains", "color_aug_hed"]

# Needed for stain unmixing
rgb_from_hed = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
hed_from_rgb = np.linalg.inv(rgb_from_hed)


def separate_stains(rgb: pyvips.Image) -> pyvips.Image:
    """Separate rgb into HED stain

    adapted from https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/color/colorconv.py#L1638
    Relevant issues:
    https://github.com/libvips/pyvips/issues/289
    https://github.com/libvips/pyvips/issues/294

    Parameters
    ----------
    rgb : pyvips.Image (uchar)
        Pyvips image in RGB colour space and uchar format (uint8) or float32/64
    Returns
    -------
    stains : pyvips.Image
        The color unmixed image from RGB to HED as float32/64.

    """

    # convert uint8 to [0,1] float 32
    if rgb.format == "uchar":
        rgb = (
            rgb.cast("float") / 255
        )  # alternatively: hed.colourspace('scrgb') gives darker colours
    elif rgb.format not in ("float", "double"):
        raise TypeError(
            "format must be one of uchar [0,255], float [0, 1], or double [0,1]"
        )

    pyvips_image = (rgb < 1e-6).ifthenelse(1e-6, rgb)  # Avoiding log artifacts
    log_adjust = np.log(1e-6)  # used to compensate the sum above
    stains = pyvips_image.log() / log_adjust

    stains = stains.recomb(hed_from_rgb.T.tolist())
    stains = (stains < 0).ifthenelse(0, stains)
    return stains


def combine_stains(hed: pyvips.Image) -> pyvips.Image:
    """Combine stains from HED to RGB

    adapted from https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/color/colorconv.py#L1638
    Relevant issues:
    https://github.com/libvips/pyvips/issues/289
    https://github.com/libvips/pyvips/issues/294

    Parameters
    ----------
    hed : pyvips.Image
        The Image in the HED colourspace as uchar (uint8) or float32/64

    Returns
    -------
    rgb : pyvips.Image
        The color unmixed RGB image as uchar (uint8)

    """

    if hed.format == "uchar":
        hed = hed.cast("float") / 255  # alternatively: hed.colourspace('scrgb')
    elif hed.format not in ("float", "double"):
        raise TypeError(
            "format must be one of uchar [0,255], float [0, 1], or double [0,1]"
        )

    # log_adjust here is used to compensate the sum within separate_stains().
    log_adjust = -np.log(1e-6)
    log_hed = -(hed * log_adjust)

    log_rgb = log_hed.recomb(rgb_from_hed.T.tolist())

    rgb = log_rgb.exp()

    rgb = (rgb < 0).ifthenelse(0, rgb)
    rgb = (rgb > 1).ifthenelse(1, rgb)

    return (rgb * 255).cast("uchar")


def color_aug_hed(img: pyvips.Image, shift) -> pyvips.Image:
    """Stain augmentation in HED colourspace

    Perform color unmixing from RGB to HED color space.
    perturb the resulting HED channels separately and recombine the channels into RGB

    Parameters
    ----------
    img : pyvips.Image
        The image to be augmented in HED colour space
    shift : list of floats
        A list of 3 floats [h_value, e_value, d_value] by which each of the channels should be shifted


    Returns
    -------
    rgb : pyvips.Image
        The HED color augmented image

    """

    img = img.cast("float") / 255
    hed = separate_stains(img)

    # Augment the Haematoxylin channel.
    hed = hed + shift

    # Back to rgb
    rgb = combine_stains(hed)

    return rgb
