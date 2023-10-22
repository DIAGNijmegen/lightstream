"""
adapted from
https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/geometric/transforms.py
The transformations are basically the same as albumentations, replacing numpy operations with pyvips wherever needed

"""

import random
import pyvips

# we need both albumentations functional and our custom one

from . import functional as FT

from ...core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = [
    "HEDShift",
    "HueSaturationValue",
    "RandomBrightnessContrast",
    "RandomGamma",
    "GaussNoise",
]


class HueSaturationValue(ImageOnlyTransform):
    """Randomly change hue, saturation and value of the input image.

    Args:
        hue_shift_limit ((int, int) or int): range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
        sat_shift_limit ((int, int) or int): range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
        val_shift_limit ((int, int) or int): range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        always_apply=False,
        p=0.5,
    ):
        super(HueSaturationValue, self).__init__(always_apply, p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0, **params):
        if not image.bands == 3 and not image.bands == 1:
            raise TypeError(
                "HueSaturationValue transformation expects 1-channel or 3-channel images."
            )
        return FT.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return {
            "hue_shift": random.uniform(
                self.hue_shift_limit[0], self.hue_shift_limit[1]
            ),
            "sat_shift": random.uniform(
                self.sat_shift_limit[0], self.sat_shift_limit[1]
            ),
            "val_shift": random.uniform(
                self.val_shift_limit[0], self.val_shift_limit[1]
            ),
        }

    def get_transform_init_args_names(self):
        return ("hue_shift_limit", "sat_shift_limit", "val_shift_limit")


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=True,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply(self, img, alpha=1.0, beta=0.0, **params):
        return FT.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)

    def get_params(self):
        return {
            "alpha": 1.0
            + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0
            + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit", "brightness_by_max")


class RandomGamma(ImageOnlyTransform):
    """
    Args:
        gamma_limit (float or (float, float)): If gamma_limit is a single float value,
            the range will be (-gamma_limit, gamma_limit). Default: (80, 120).
        eps: Deprecated.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5):
        super(RandomGamma, self).__init__(always_apply, p)
        self.gamma_limit = to_tuple(gamma_limit)
        self.eps = eps

    def apply(self, img, gamma=1, **params):
        return FT.gamma_transform(img, gamma=gamma)

    def get_params(self):
        return {
            "gamma": random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0
        }

    def get_transform_init_args_names(self):
        return ("gamma_limit", "eps")


class GaussNoise(ImageOnlyTransform):
    """Apply gaussian noise to the input image.

    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        per_channel (bool): if set to True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels. Default: True
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        var_limit=(10.0, 50.0),
        mean=0,
        per_channel=True,
        always_apply=False,
        p=0.5,
    ):
        super(GaussNoise, self).__init__(always_apply, p)
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(
                    type(var_limit)
                )
            )

        self.mean = mean
        self.per_channel = per_channel

    def apply(self, img, gauss=None, **params):
        return FT.gauss_noise(img, gauss=gauss)

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var**0.5

        if self.per_channel:
            gauss_1 = pyvips.Image.gaussnoise(
                image.width, image.height, sigma=sigma, mean=self.mean
            )
            gauss_2 = pyvips.Image.gaussnoise(
                image.width, image.height, sigma=sigma, mean=self.mean
            )
            gauss_3 = pyvips.Image.gaussnoise(
                image.width, image.height, sigma=sigma, mean=self.mean
            )
            gauss = gauss_1.bandjoin([gauss_2, gauss_3])
        else:
            gauss = pyvips.Image.gaussnoise(
                image.width, image.height, sigma=sigma, mean=self.mean
            )

        return {"gauss": gauss}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("var_limit", "per_channel", "mean")


class HEDShift(ImageOnlyTransform):
    """todo: Add masking support

    perhaps use this to add mask as param?
    https://stackoverflow.com/questions/76125174/custom-mask-dependent-augmentation-in-albumentations
    mask = (pyvips_image >= 250).bandand()
    plt.imshow(mask.numpy())
    plt.show()

    result = (mask).ifthenelse(
        mask, color_aug_hed(pyvips_image, shift=[0.0, 0.05, 0.0])
    )

    """

    def __init__(
        self,
        h_shift_limit: float = 0.03,
        e_shift_limit: float = 0.03,
        d_shift_limit: float = 0.03,
        always_apply: bool = False,
        p: float = 0.5,
        **params
    ):
        super().__init__(always_apply, p, **params)
        self.h_shift_limit = to_tuple(h_shift_limit)
        self.e_shift_limit = to_tuple(e_shift_limit)
        self.d_shift_limit = to_tuple(d_shift_limit)

    def apply(self, image, h_shift=0, e_shift=0, d_shift=0, **params):
        shift = [h_shift, e_shift, d_shift]
        return FT.color_aug_hed(image, shift)

    def get_params(self):
        return {
            "h_shift": random.uniform(self.h_shift_limit[0], self.h_shift_limit[1]),
            "e_shift": random.uniform(self.e_shift_limit[0], self.e_shift_limit[1]),
            "d_shift": random.uniform(self.d_shift_limit[0], self.d_shift_limit[1]),
        }

    def get_transform_init_args_names(self):
        return "h_shift_limit", "e_shift_limit", "d_shift_limit"
