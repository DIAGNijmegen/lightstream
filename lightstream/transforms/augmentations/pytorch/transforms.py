import torch
import pyvips
import numpy as np
from ...core.transforms_interface import BasicTransform, ImageOnlyTransform
from . import functional as FT

__all__ = [
    "Normalize",
    "ToDtype",
    "ToTensor"
]
class Normalize(ImageOnlyTransform):
    """Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`

    Args:
        mean (list of float): mean values
        std  (list of float): std values
        max_pixel_value (float): maximum possible pixel value

    Targets:
        image

    Image types:
        float32

    """

    def __init__(self, mean: list, std: list, always_apply: bool = False, p: float = 1.0, **params):
        super().__init__(always_apply, p, **params)
        self.mean = mean
        self.std = std
    def apply(self, img, **params):
        return FT.normalize(img, self.mean, self.std)

    def get_transform_init_args_names(self):
        return "mean", "std",


class ToDtype(ImageOnlyTransform):
    """ Cast the image to another data type
    It can also take into account proper scaling to the appropriate range when casting from or to uchar and floats.
    Going from uchar to float, it will first cast to float, and then divide by 255. The other way around,
    The image is first multiplied by 255, and then cast to uchar

    Parameters
    ----------
    dtype : str
        The datatype to transfer to, can be one of uchar, float, or double
    scale : bool
        Whether to scale the data to the appropriate range when casting from/to float and uchar (uint8)
    """
    def __init__(self, dtype: str, scale: bool, always_apply: bool = False, p: float = 1.0, **params):
        super().__init__(always_apply, p, **params)
        self.dtype = dtype
        self.scale = scale
    def apply(self, img, **params):
        return FT.to_dtype(img, self.dtype, self.scale)

    def get_transform_init_args_names(self):
        return "dtype", "scale",

class ToTensor(BasicTransform):
    """Convert image and mask to `torch.Tensor`. The numpy `HWC` image is converted to pytorch `CHW` tensor.
    If the image is in `HW` format (grayscale image), it will be converted to pytorch `HW` tensor.
    This is a simplified and improved version of the old `ToTensor`
    transform (`ToTensor` was deprecated, and now it is not present in Albumentations. You should use `ToTensorV2`
    instead).

    Args:
        transpose_mask (bool): If True and an input mask has three dimensions, this transform will transpose dimensions
            so the shape `[height, width, num_channels]` becomes `[num_channels, height, width]`. The latter format is a
            standard format for PyTorch Tensors. Default: False.
        always_apply (bool): Indicates whether this transformation should be always applied. Default: True.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(
        self, transpose_mask: bool = False, always_apply: bool = True, p: float = 1.0
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
        }

    def apply(self, img: pyvips.Image, **params) -> torch.tensor:  # skipcq: PYL-W0613
        img = img.numpy()
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(
        self, mask: pyvips.Image, **params
    ) -> torch.tensor:  # skipcq: PYL-W0613
        mask = mask.numpy()
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    def apply_to_masks(self, masks, **params) -> list:
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}
