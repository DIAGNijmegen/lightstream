import random
import warnings

from . import functional as FT
from albumentations.augmentations.crops import functional as F

from ...core.transforms_interface import DualTransform


__all__ = ["RandomCrop", "CropOrPad"]


class RandomCrop(DualTransform):
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        return FT.random_crop(img, self.height, self.width, h_start, w_start)

    def get_params(self):
        return {"h_start": random.random(), "w_start": random.random()}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_random_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_random_crop(keypoint, self.height, self.width, **params)

    def get_transform_init_args_names(self):
        return ("height", "width")


class CropOrPad(DualTransform):
    """Crop or pad a given image to a specified width/height

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        direction (str): Direction to place image within width/height, can be one of
            "centre" (default), "north", "east", "south", "west",
            "north-east", "south-east", "south-west", "north-west"
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, direction="centre", always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.direction = direction

    def apply(self, img, **params):
        return FT.pad_or_crop(
            img,
            self.width,
            self.height,
            background=[255, 255, 255],
            direction=self.direction
        )

    def apply_to_mask(self, img, **params):
        return FT.pad_or_crop(
            img,
            self.width,
            self.height,
            background=[0, 0, 0],
            direction=self.direction
        )

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError()

    def apply_to_keypoint(self, keypoint, **params):
        "This is not implemented yet, not needed for this project"
        raise NotImplementedError()

    def get_transform_init_args_names(self):
        return ("height", "width", "direction")
