"""
adapted from
https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/geometric/transforms.py
The transformations are basically the same as albumentations, replacing numpy operations with pyvips wherever needed

"""

import pyvips
import random
import numpy as np

# we need both albumentations functional and our custom one

from . import functional as FT

from ...core.transforms_interface import (
    ImageOnlyTransform,
    to_tuple
)

__all__ = [
    "HEDShift"
]


class HEDShift(ImageOnlyTransform):
    """ todo: Add masking support"""
    def __init__(self, h_shift_limit: float = 0.03, e_shift_limit: float = 0.03, d_shift_limit: float = 0.03,
                 always_apply: bool = False, p: float = 0.5, **params):
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
        return ("h_shift_limit", "e_shift_limit", "d_shift_limit")

