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
from enum import Enum
from albumentations.augmentations.geometric import functional as F
from albumentations.augmentations.functional import bbox_from_mask
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox

from . import functional as FT

from ...core.transforms_interface import BoxInternalType, DualTransform, KeypointInternalType, ImageColorType

__all__ = ["VerticalFlip", "HorizontalFlip", "Flip", "Transpose", "ElasticTransform", "RandomRotate90", "PadIfNeeded"]


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

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
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

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
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

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
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

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
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

    def apply(self, img, random_state=None, interpolation=pyvips.Interpolate.new("bilinear"), **params):
        return FT.elastic_transform(img, self.alpha, self.sigma, interpolation, self.value, self.same_dxdy)

    def apply_to_mask(self, img, random_state=None, **params):
        return FT.elastic_transform(
            img, self.alpha, self.sigma, pyvips.Interpolate.new("nearest"), self.mask_value, self.same_dxdy
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
            mask, self.alpha, self.sigma, pyvips.Interpolate.new("nearest"), self.mask_value, self.same_dxdy
        )
        bbox_returned = bbox_from_mask(mask)
        bbox_returned = F.normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned

    def get_params(self):
        return {"random_state": random.randint(0, 10000)}

    def get_transform_init_args_names(self):
        return ("alpha", "sigma", "interpolation", "value", "mask_value", "same_dxdy")


class PadIfNeeded(DualTransform):
    """Pad side of the image / max if side is less than desired number.

    Args:
        min_height (int): minimal result image height.
        min_width (int): minimal result image width.
        pad_height_divisor (int): if not None, ensures image height is dividable by value of this argument.
        pad_width_divisor (int): if not None, ensures image width is dividable by value of this argument.
        position (Union[str, PositionType]): Position of the image. should be PositionType.CENTER or
            PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
            Default: PositionType.CENTER.
        border_mode (pyvips enums extend): pyvips border mode.
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value for mask if border_mode is pyvips.enums.Extend.constant.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32
    """

    class PositionType(Enum):
        CENTER = "centre"
        TOP_LEFT = "north-west"
        TOP_RIGHT = "north-east"
        BOTTOM_LEFT = "south-west"
        BOTTOM_RIGHT = "south-east"

    def __init__(
        self,
        min_height: int | None = 1024,
        min_width: int | None = 1024,
        pad_height_divisor: int | None = None,
        pad_width_divisor: int | None = None,
        position: PositionType | str = PositionType.CENTER,
        border_mode: int = pyvips.enums.Extend.BACKGROUND,
        value: ImageColorType | None = None,
        mask_value: ImageColorType | None = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        if (min_height is None) == (pad_height_divisor is None):
            raise ValueError("Only one of 'min_height' and 'pad_height_divisor' parameters must be set")

        if (min_width is None) == (pad_width_divisor is None):
            raise ValueError("Only one of 'min_width' and 'pad_width_divisor' parameters must be set")

        super(PadIfNeeded, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.position = PadIfNeeded.PositionType(position)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if self.min_height is not None:
            if rows < self.min_height:
                h_pad_top = int((self.min_height - rows) / 2.0)
                h_pad_bottom = self.min_height - rows - h_pad_top
            else:
                h_pad_top = 0
                h_pad_bottom = 0
        else:
            pad_remained = rows % self.pad_height_divisor
            pad_rows = self.pad_height_divisor - pad_remained if pad_remained > 0 else 0

            h_pad_top = pad_rows // 2
            h_pad_bottom = pad_rows - h_pad_top

        if self.min_width is not None:
            if cols < self.min_width:
                w_pad_left = int((self.min_width - cols) / 2.0)
                w_pad_right = self.min_width - cols - w_pad_left
            else:
                w_pad_left = 0
                w_pad_right = 0
        else:
            pad_remainder = cols % self.pad_width_divisor
            pad_cols = self.pad_width_divisor - pad_remainder if pad_remainder > 0 else 0

            w_pad_left = pad_cols // 2
            w_pad_right = pad_cols - w_pad_left

        (h_pad_top, h_pad_bottom, w_pad_left, w_pad_right) = self.__update_position_params(
            h_top=h_pad_top, h_bottom=h_pad_bottom, w_left=w_pad_left, w_right=w_pad_right
        )

        new_width = cols + w_pad_left + w_pad_right
        new_height = rows + h_pad_top + h_pad_bottom

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
                "new_height": new_height,
                "new_width": new_width,
            }
        )
        return params

    def apply(self, img: pyvips.Image, new_width: int = 0, new_height: int = 0, **params) -> pyvips.Image:
        return FT.pad_with_params(
            img,
            direction=self.position.value,
            width=new_width,
            height=new_height,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(self, img: pyvips.Image, new_width: int = 0, new_height: int = 0, **params) -> pyvips.Image:
        return FT.pad_with_params(
            img,
            direction=self.position.value,
            width=new_width,
            height=new_height,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        rows: int = 0,
        cols: int = 0,
        **params
    ) -> BoxInternalType:
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        **params
    ) -> KeypointInternalType:
        x, y, angle, scale = keypoint[:4]
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return (
            "min_height",
            "min_width",
            "pad_height_divisor",
            "pad_width_divisor",
            "border_mode",
            "value",
            "mask_value",
        )

    def __update_position_params(
        self, h_top: int, h_bottom: int, w_left: int, w_right: int
    ) -> tuple[int, int, int, int]:
        if self.position == PadIfNeeded.PositionType.TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            h_top = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            h_top = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            h_bottom = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            h_bottom = 0
            w_right = 0

        return h_top, h_bottom, w_left, w_right
