import random
import warnings
import numpy as np

from . import functional as FT
from ...core.transforms_interface import (
    ImageOnlyTransform,
    to_tuple,
    ScaleIntType,
    ScaleFloatType,
)

__all__=["GaussianBlur"]

class GaussianBlur(ImageOnlyTransform):
    """Blur the input image using a Gaussian filter with a random kernel size.

    This implementation is not faithful to albumentation's gaussian blur. CV2 libraries have a predefined kernel
    or sigma, or calculate either the radius/sigma from the given kernel size. Pyvips defines an amplitude that
    determines the accuracy of the mask: Given a certain accuracy, the kernel size may change given varying levels
    of sigma

    In short: opencv sets the speed of the algorithm with a predefined kernel size, and the accuracy of the mask
    differs with sigma. Pyvips sets the accuracy, and the kernel size will change with varying sigma.

    Reference: https://github.com/libvips/libvips/discussions/3038

    Args:
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        min_amplitude (float): must be in range [0,inf)
            accuracy of the mask, implicitly determines radius/kernel size. Lower values typically lead to higher accuracy
            and larger kernels, at the cost of speed. Recommended to keep this at default of 0.2
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        sigma_limit: ScaleFloatType = 0,
        min_amplitude: float = 0.2,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.sigma_limit = to_tuple(sigma_limit if sigma_limit is not None else 0, 0)
        self.min_amplitude = min_amplitude

    def apply(
        self, img: np.ndarray, min_amplitude: float = 0.2, sigma: float = 0, **params
    ) -> np.ndarray:
        return FT.gaussian_blur(img, sigma=sigma, min_amplitude=self.min_amplitude)

    def get_params(self) -> dict[str, float]:
        return {"sigma": random.uniform(*self.sigma_limit)}

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "sigma_limit", "min_amplitude"
