"""

Functional representations of the augmentations. Most will function as functional pyvips aliases.

"""

import pyvips


__all__ = [
    "vflip",
    "hflip",
    "random_flip",
    "transpose"
]

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
        The pyvips image as uchar (uint8) dtype
    code : int
        Mode for flipping the image, follows cv2.flip() coding
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

