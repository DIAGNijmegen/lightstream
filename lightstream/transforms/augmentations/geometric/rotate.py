import pyvips
import random
import numpy as np

# we need both albumentations functional and our custom one
from albumentations.augmentations.geometric import functional as F
import pyvips

from ...core.transforms_interface import DualTransform


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: pyvips.Image, factor: int = 0, **params) -> pyvips.Image:
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        img = [img.rot90, img.rot180, img.rot270]
        return img[factor]()

    def get_params(self):
        # Random int in the range [0, 3]
        return {"factor": random.randint(0, 3)}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return F.bbox_rot90(bbox, factor, **params)

    def apply_to_keypoint(self, keypoint, factor=0, **params):
        return F.keypoint_rot90(keypoint, factor, **params)

    def get_transform_init_args_names(self):
        return ()
