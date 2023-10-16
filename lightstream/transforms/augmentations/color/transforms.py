"""
adapted from
https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/geometric/transforms.py
The transformations are basically the same as albumentations, replacing numpy operations with pyvips wherever needed

"""

import pyvips
import random

# we need both albumentations functional and our custom one
from albumentations.augmentations.geometric import functional as F

from . import functional as FT

from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    KeypointInternalType,
)


