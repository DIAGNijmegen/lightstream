import cv2
import math
import pyvips
import random

# we need both albumentations functional and our custom one
from albumentations.augmentations.geometric import functional as F
from albumentations.augmentations.crops import functional as FCrops
from ..crops import functional as FTCrops
from . import functional as FT

from ...core.transforms_interface import DualTransform, to_tuple

__all__ = ["RandomRotate90", "Rotate"]


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


class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        crop_border (bool): If True would make a largest possible crop within rotated image
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        limit=90,
        interpolation=pyvips.Interpolate.new("bilinear"),
        value=[255, 255, 255],
        mask_value=[0, 0, 0],
        rotate_method="largest_box",
        crop_border=False,
        always_apply=False,
        p=0.5,
    ):
        super(Rotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method
        self.crop_border = crop_border

        if rotate_method not in ["largest_box", "ellipse"]:
            raise ValueError(f"Rotation method {self.rotate_method} is not valid.")

    def apply(
        self,
        img,
        angle=0,
        interpolation=pyvips.Interpolate.new("bilinear"),
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        **params,
    ):
        print("angle value:", angle)
        print(f"image width and height before transform: {img.width}, {img.height}")

        img_out = FT.rotate(img, angle, interpolation, self.value)
        print(
            f"image width and height after transform: {img_out.width}, {img_out.height}"
        )

        if self.crop_border:
            img_out = FTCrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_mask(
        self, img, angle=0, x_min=None, x_max=None, y_min=None, y_max=None, **params
    ):
        img_out = FT.rotate(
            img, angle, pyvips.Interpolate.new("nearest"), self.mask_value
        )
        if self.crop_border:
            img_out = FTCrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_bbox(
        self,
        bbox,
        angle=0,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        cols=0,
        rows=0,
        **params,
    ):
        bbox_out = F.bbox_rotate(bbox, angle, self.rotate_method, rows, cols)
        if self.crop_border:
            bbox_out = FCrops.bbox_crop(
                bbox_out, x_min, y_min, x_max, y_max, rows, cols
            )
        return bbox_out

    def apply_to_keypoint(
        self,
        keypoint,
        angle=0,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        cols=0,
        rows=0,
        **params,
    ):
        keypoint_out = F.keypoint_rotate(keypoint, angle, rows, cols, **params)
        if self.crop_border:
            keypoint_out = FCrops.crop_keypoint_by_coords(
                keypoint_out, (x_min, y_min, x_max, y_max)
            )
        return keypoint_out

    @staticmethod
    def _rotated_rect_with_max_area(h, w, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        degrees), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.

        Code from: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """

        angle = math.radians(angle)
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # it is sufficient to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (
                (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
            )
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

        return dict(
            x_min=max(0, int(w / 2 - wr / 2)),
            x_max=min(w, int(w / 2 + wr / 2)),
            y_min=max(0, int(h / 2 - hr / 2)),
            y_max=min(h, int(h / 2 + hr / 2)),
        )

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict[str,]) -> dict[str,]:
        out_params = {"angle": random.uniform(self.limit[0], self.limit[1])}
        if self.crop_border:
            h, w = params["image"].height, params["image"].width
            out_params.update(
                self._rotated_rect_with_max_area(h, w, out_params["angle"])
            )
        return out_params

    def get_transform_init_args_names(self):
        return (
            "limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
            "rotate_method",
            "crop_border",
        )
