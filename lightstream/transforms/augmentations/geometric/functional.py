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


def elastic_deformation(image: pyvips.Image, sigma: int=32, alpha:int=4):
    """ Apply elastic transformation on the image



    Parameters
    ----------
    image
    sigma
    alpha

    Returns
    -------

    """
    width, height, channels = image.width, image.height, image.bands

    # Create a random displacement field, pyvips does not have uniform
    # instead, use a Gaussian and convert using Box-Muller inverse
    z1 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)
    z2 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)

    # Compute box-muller inverse to get approximate uniform values
    dx = (-(z1*z1 + z2*z2) / 2).exp()
    dx = (2 * dx - 1).gaussblur(sigma) * alpha

    dy = (z1 / z2).atan()
    dy = (2 * dy - 1).gaussblur(sigma) * alpha

    grid = pyvips.Image.xyz(image.width, image.height)
    new_coords = grid + dx.bandjoin([dy])

    image = image.mapim(new_coords, interpolate=pyvips.Interpolate.new('bilinear'), background=[255, 255, 255])

    return image

