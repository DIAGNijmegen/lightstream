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


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: pyvips.Image, **params) -> pyvips.Image:
        return FT.vflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_vflip(bbox, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        return F.keypoint_vflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: pyvips.Image, **params) -> pyvips.Image:
        return FT.hflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_hflip(bbox, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        return F.keypoint_hflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class Flip(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: pyvips.Image, d: int = 0, **params) -> pyvips.Image:
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        return FT.random_flip(img, d)

    def get_params(self):
        # Random int in the range [-1, 1]
        return {"d": random.randint(-1, 1)}

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_flip(bbox, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        return F.keypoint_flip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class Transpose(DualTransform):
    """Transpose the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: pyvips.Image, **params) -> pyvips.Image:
        return FT.transpose(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_transpose(bbox, 0, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        return F.keypoint_transpose(keypoint)

    def get_transform_init_args_names(self):
        return ()
