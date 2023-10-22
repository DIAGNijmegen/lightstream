"""

Functional representations of the augmentations. Most will function as functional pyvips aliases.

"""

import pyvips
import numpy as np
from warnings import warn
from ..utils import MAX_VALUES_BY_DTYPE

__all__ = [
    "separate_stains",
    "combine_stains",
    "color_aug_hed",
    "shift_hsv",
    "brightness_contrast_adjust",
    "gamma_transform",
    "gauss_noise",
]

# Needed for stain unmixing
rgb_from_hed = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
hed_from_rgb = np.linalg.inv(rgb_from_hed)


def _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift):
    # https://github.com/libvips/libvips/discussions/2486
    # hsv is all in the range of [0,255]
    img = img.colourspace("hsv")
    # print("img before lut", img.numpy())
    lut = pyvips.Image.identity(bands=img.bands, ushort=(img.format == "ushort"))

    hue_shift, sat_shift, val_shift = int(hue_shift), int(sat_shift), int(val_shift)

    # In pyvips, hue is encoded in [0,255], uchar will clip to [0,255]
    lut += [hue_shift, sat_shift, val_shift]

    new_h = img[0].maplut(lut[0].cast("uchar"))
    new_s = img[1].maplut(lut[1].cast("uchar"))
    new_v = img[2].maplut(lut[2].cast("uchar"))

    hsv = new_h.bandjoin([new_s, new_v])

    img = hsv.colourspace("srgb")

    return img.colourspace("srgb")


def _shift_hsv_uint8_2(img, hue_shift, sat_shift, val_shift):
    # Deprecate?
    # https://github.com/libvips/libvips/discussions/2486
    # hsv is all in the range of [0,255]

    img = img.colourspace("hsv")

    # Build a lut for this?
    img += [int(hue_shift), int(sat_shift), int(val_shift)]
    # casting to uchar will clip to [0, 255]
    img = img.cast("uchar")
    return img.colourspace("srgb")


def _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift):
    # Assuming that hsv shifts are still in uchar pixels
    dtype = img.format
    hue_shift, sat_shift, val_shift = (
        hue_shift / 255.0,
        sat_shift / 255.0,
        val_shift / 255,
    )

    img = img.colourspace("hsv")
    img += [hue_shift, sat_shift, val_shift]
    img = (img > 1).ifthenelse(1, img)
    img = (img < 0).ifthenelse(0, img)

    return img.colourspace("srgb")


def shift_hsv(img, hue_shift, sat_shift, val_shift):
    if hue_shift == 0 and sat_shift == 0 and val_shift == 0:
        return img

    # if grayscale
    if img.bands == 1:
        if hue_shift != 0 or sat_shift != 0:
            hue_shift = 0
            sat_shift = 0
            warn(
                "HueSaturationValue: hue_shift and sat_shift are not applicable to grayscale image. "
                "Set them to 0 or use RGB image"
            )

    if img.format == "uchar":
        img = _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift)
    else:
        img = _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift)

    return img


def _brightness_contrast_adjust_non_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = img.format
    img = img.cast("float")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * img.avg()

    img = (img > 1.0).ifthenelse(1.0, img)
    img = (img < 0.0).ifthenelse(0.0, img)

    return img


def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = "uchar"

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    lut = pyvips.Image.identity(bands=img.bands, ushort=(img.format == "ushort"))
    lut = lut.cast("float")

    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += (alpha * beta) * img.avg()

    # maplut on an image returns a histogram, so do lut by band, which is still an image for some reason.
    new_r = img[0].maplut(lut[0].cast(img.format))
    new_g = img[1].maplut(lut[1].cast(img.format))
    new_b = img[2].maplut(lut[2].cast(img.format))
    return new_r.bandjoin([new_g, new_b])


def gauss_noise(image, gauss):
    # Perhaps will not work well when images are float32 from the start
    dtype = image.format
    image = image.cast("float")
    image = image + gauss
    return (image + gauss).cast(dtype)


def gamma_transform(img, gamma):
    """
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    else:
        img = np.power(img, gamma)
    """
    return img.gamma(exponent=gamma)


def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):
    if img.format == "uchar":
        return _brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)

    return _brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)


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

    # Casting to uchar would already clip this, so perhaps remove?
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
