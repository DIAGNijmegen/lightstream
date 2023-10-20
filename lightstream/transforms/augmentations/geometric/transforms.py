"""
adapted from
https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/geometric/transforms.py
The transformations are basically the same as albumentations, replacing numpy operations with pyvips wherever needed

"""

import pyvips
import random

import numpy as np
import cv2

# we need both albumentations functional and our custom one
from albumentations.augmentations.geometric import functional as F
from albumentations.augmentations.functional import bbox_from_mask

from . import functional as FT

from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    KeypointInternalType,
)

__all__ = [
    "VerticalFlip",
    "HorizontalFlip",
    "Flip",
    "Transpose",
    "ElasticTransform",
    "RandomRotate90",
]


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


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        return FT.rot90(img, factor)

    def get_params(self):
        # Random int in the range [0, 3]
        return {"factor": random.randint(0, 3)}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return F.bbox_rot90(bbox, factor, **params)

    def apply_to_keypoint(self, keypoint, factor=0, **params):
        return F.keypoint_rot90(keypoint, factor, **params)

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


class ElasticTransform(DualTransform):
    """Elastic deformation of images as described in [Simard2003] .

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Args:
        alpha (float): Magnitude/intensity of the displacement
        sigma (float): Gaussian filter parameter.
        interpolation (Pyvips Interpolate): Interpolation object that is used to specify the interpolation algorithm. Should be one of:
            pyvips.Interpolate.new(bilinear), pyvips.Interpolate.new(cubic), pyvips.Interpolate.new(nearest).
            Default: pyvips.Interpolate.new("bilinear").
        value (list of ints,
                list of floats): padding value for the background, applied to the image.
        mask_value (list of ints,
                    list of float): padding value for the background, applied to the mask.
        same_dxdy (boolean): DOES NOT WORK FOR NOW: Whether to use same random generated shift for x and y.

    Targets:
        image, mask, bbox

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        alpha: float = 1.0,
        sigma: float = 50.0,
        interpolation: pyvips.Interpolate = pyvips.Interpolate.new("bilinear"),
        value: list = None,
        mask_value: list = None,
        always_apply: bool = False,
        same_dxdy: bool = False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        print("Elastic transform is only supported for image-only datasets, do not use with bounding boxes/masks!")
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.value = [255, 255, 255] if value is None else value
        self.mask_value = [0, 0, 0] if mask_value is None else mask_value
        self.same_dxdy = same_dxdy

    def apply(
        self,
        img,
        random_state=None,
        interpolation=pyvips.Interpolate.new("bilinear"),
        **params
    ):
        return FT.elastic_transform(
            img, self.alpha, self.sigma, interpolation, self.value, self.same_dxdy
        )

    def apply_to_mask(self, img, random_state=None, **params):
        return FT.elastic_transform(
            img,
            self.alpha,
            self.sigma,
            pyvips.Interpolate.new("nearest"),
            self.mask_value,
            self.same_dxdy,
        )

    def apply_to_bbox(self, bbox, random_state=None, **params):
        """Does not work for now: All numpy arrays so will crash when used on pyvips functions"""
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = F.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = F.elastic_transform(
            mask,
            self.alpha,
            self.sigma,
            pyvips.Interpolate.new("nearest"),
            self.mask_value,
            self.same_dxdy,
        )
        bbox_returned = bbox_from_mask(mask)
        bbox_returned = F.normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned

    def get_params(self):
        return {"random_state": random.randint(0, 10000)}

    def get_transform_init_args_names(self):
        return ("alpha", "sigma", "interpolation", "value", "mask_value", "same_dxdy")
