"""

Functional representations of the augmentations. Most will function as functional pyvips aliases.

"""

import pyvips
import numpy as np

# Needed for stain unmixing
rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]])
hed_from_rgb = np.linalg.inv(rgb_from_hed)



def vflip(img: pyvips.Image) -> pyvips.Image:
    return img.flipver()


def hflip(img: pyvips.Image) -> pyvips.Image:
    return img.fliphor()


def random_flip(img: pyvips.Image, code: int) -> pyvips.Image:
    """Pyvips random flips that emulated cv2.flip

    cv2.flip(img,d) takes an additional code that specifies the flip:
    -1: both horizontal and vertical flip, 0 is vertical, 1 is horizontal


    Parameters
    ----------
    img : pyvips.Image
    code : int

    Returns
    -------
    img: pyvips.Image

    """
    if code == 1:
        img = img.fliphor()
    elif code == 0:
        img = img.flipver()
    else:
        img = img.fliphor().flipver()

    return img


def transpose(img: pyvips.Image) -> pyvips.Image:
    """Albumentation transpose an image

    Albumentations transposes by switching rows and column in numpy. This is not the same as the matrix transpose
    used in linear algebra. Instead, the outcome is a 270 degree flip. For a proper transpose, additionally
    flip the result vertically.

    Parameters
    ----------
    img: pyvips.Image

    Returns
    -------
    img: pyvips.Image

    """

    return img.rot270()


def separate_stains_pyvips(rgb: pyvips.Image):
    # adapted from https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/color/colorconv.py#L1638
    # https://github.com/libvips/pyvips/issues/289
    # https://github.com/libvips/pyvips/issues/294

    # convert uint8 to [0,1] float 32
    if rgb.format == 'uchar':
        # dividing by 255 yields faithful results to skimage
        # colourspace('scrgb') is recommended by jcupitt, but yields more intense colours

        # rgb = rgb.cast('double') / 255
        rgb = rgb.colourspace("scrgb")
    elif rgb.format not in ('float', 'double'):
        raise TypeError("format must be one of uchar [0,255], float [0, 1], or double [0,1]")

    pyvips_image = (rgb < 1E-6).ifthenelse(1E-6, rgb)  # Avoiding log artifacts
    log_adjust = np.log(1E-6)  # used to compensate the sum above
    stains = (pyvips_image.log() / log_adjust)

    stains = stains.recomb(hed_from_rgb.T.tolist())
    stains = (stains < 0).ifthenelse(0, stains)
    return stains


def combine_stains_pyvips(hed):
    if hed.format == 'uchar':
        # hed = hed.cast('float') / 255
        hed = rgb.colourspace("scrgb")
    elif hed.format not in ('float', 'double'):
        raise TypeError("format must be one of uchar [0,255], float [0, 1], or double [0,1]")

    # log_adjust here is used to compensate the sum within separate_stains().
    log_adjust = -np.log(1E-6)
    log_hed = -(hed * log_adjust)

    log_rgb = log_hed.recomb(rgb_from_hed.T.tolist())

    rgb = log_rgb.exp()

    rgb = (rgb < 0).ifthenelse(0, rgb)
    rgb = (rgb > 1).ifthenelse(1, rgb)
    return rgb