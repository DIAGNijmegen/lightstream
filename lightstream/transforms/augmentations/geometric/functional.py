"""

Functional representations of the augmentations. Most will function as functional pyvips aliases.

"""

import pyvips

__all__ = ["vflip", "hflip", "random_flip", "rot90", "transpose", "elastic_transform"]


def vflip(img: pyvips.Image) -> pyvips.Image:
    return img.flipver()


def hflip(img: pyvips.Image) -> pyvips.Image:
    return img.fliphor()


def rot90(img: pyvips.Image, factor: int) -> pyvips.Image:
    """

    Parameters
    ----------
    img : pyvips.Image
        The pyvips image as uchar (uint8) dtype
    factor : int
        Number of times the input will be rotated by 90 degrees.
    Returns
    -------
    img: pyvips.Image

    """
    # only 4 choices for 90 degrees, regardless of the factor value
    factor = factor % 4
    options = [img, img.rot90, img.rot180, img.rot270]
    print(factor)
    result = options[factor] if factor == 0 else options[factor]()
    return result


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
        The pyvips image as uchar (uint8) dtype
    Returns
    -------
    img: pyvips.Image

    """

    return img.rot270()


def elastic_transform(
    img: pyvips.Image,
    alpha: float = 1.0,
    sigma: float = 50.0,
    interpolation: pyvips.Interpolate = pyvips.Interpolate.new("bilinear"),
    background: list = [255, 255, 255],
    same_dxdy: bool = False,
) -> pyvips.Image:
    """Apply elastic transformation on the image

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Parameters
    ----------
    image : Pyvips Image object
        The image on which to apply elastic deformation
    sigma : int
        Elasticity coefficient : standard deviation for Gaussian blur, controls smoothness, higher is smoother
    alpha : int
        Scaling  factor that controls the intensity (magnitude) of the displacements
    interpolation : pyvips.Interpolate
        The interpolation to use, can be one of bilinear, cubic, linear, nearest
    same_dx_dy : bool
        DOES NOTHING FOR NOW Whether to use the same displacement for both x and y directions.

    Returns
    -------
    img: pyvips.Image

    """
    width, height, channels = img.width, img.height, img.bands

    # Create a random displacement field, pyvips does not have uniform sampling (yet)
    # instead, use a Gaussian and convert using Box-Muller inverse
    z1 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)
    z2 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)

    # Compute box-muller inverse to get approximate uniform values
    dx = (-(z1 * z1 + z2 * z2) / 2).exp()
    dx = (2 * dx - 1).gaussblur(sigma) * alpha

    dy = (z1 / z2).atan()
    dy = (2 * dy - 1).gaussblur(sigma) * alpha

    grid = pyvips.Image.xyz(img.width, img.height)
    new_coords = grid + dx.bandjoin([dy])

    image = img.mapim(new_coords, interpolate=interpolation, background=background)

    return image
