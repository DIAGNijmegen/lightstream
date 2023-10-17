import albumentations as A
import numpy as np
import pyvips
import matplotlib.pyplot as plt
from PIL import Image
from lightstream.transforms import Compose, HorizontalFlip, VerticalFlip, HEDShift
from lightstream.transforms import color_aug_hed
import os
import cv2
import random

random.seed(0)
np.random.seed(0)


def draw_grid(image: pyvips.Image, grid_size):
    # Draw grid lines
    for i in range(0, image.width, grid_size):
        image = image.draw_line([0, 0, 0], i, 0, i, image.height)
    for j in range(0, image.height, grid_size):
        image = image.draw_line([0, 0, 0], 0, j, image.width, j)

    return image


def elastic_deformation(image: pyvips.Image, sigma: int = 50, alpha: int = 1):
    width, height, channels = image.width, image.height, image.bands

    # Create a random displacement field, pyvips does not have uniform
    # instead, use a Gaussian and convert using Box-Muller inverse
    z1 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)
    z2 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)

    # Compute box-muller inverse to get approximate uniform values
    dx = (-(z1 * z1 + z2 * z2) / 2).exp()
    dx = (2 * dx - 1).gaussblur(sigma) * alpha

    dy = (z1 / z2).atan()
    dy = (2 * dy - 1).gaussblur(sigma) * alpha

    grid = pyvips.Image.xyz(image.width, image.height)
    new_coords = grid + dx.bandjoin([dy])

    image = image.mapim(
        new_coords,
        interpolate=pyvips.Interpolate.new("bilinear"),
        background=[255, 255, 255],
    )

    return image


if __name__ == "__main__":
    image = np.array(Image.open("../images/he.jpg"))
    pyvips_image = pyvips.Image.new_from_array(image)
    print(pyvips_image.width, pyvips_image.height, pyvips_image.bands)
    plt.imshow(pyvips_image.numpy())
    plt.show()

    mask = (pyvips_image >= 250).bandand()
    plt.imshow(mask.numpy())
    plt.show()

    result = (mask).ifthenelse(
        mask, color_aug_hed(pyvips_image, shift=[0.0, 0.05, 0.0])
    )
    plt.imshow(result.numpy())
    plt.show()
