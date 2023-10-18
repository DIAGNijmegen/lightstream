import numpy as np
import pyvips
import random
import time
import cv2

random.seed(0)
np.random.seed(0)


def draw_grid(image: pyvips.Image, grid_size):
    # Draw grid lines
    for i in range(0, image.width, grid_size):
        image = image.draw_line([0, 0, 0], i, 0, i, image.height)
    for j in range(0, image.height, grid_size):
        image = image.draw_line([0, 0, 0], 0, j, image.width, j)

    return image


def elastic_transform(
    img: pyvips.Image,
    alpha: float = 1.0,
    sigma: float = 50.0,
    interpolation: pyvips.Interpolate = pyvips.Interpolate.new("bilinear"),
    background: list = [255, 255, 255],
    same_dxdy: bool = False,
) -> pyvips.Image:
    """Apply elastic transformation on the image

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Parameters
    ----------
    image : Pyvips Image object
        The image on which to apply elastic deformation
    sigma : int
        Elasticity coefficient : standard deviation for Gaussian blur, controls smoothness, higher is smoother
    alpha : int
        Scaling  factor that controls the intensity (magnitude) of the displacements
    interpolation : pyvips.Interpolate
        The interpolation to use, can be one of bilinear, cubic, linear, nearest
    same_dx_dy : bool
        DOES NOTHING FOR NOW Whether to use the same displacement for both x and y directions.

    Returns
    -------
    img: pyvips.Image

    """
    width, height, channels = img.width, img.height, img.bands

    # Create a random displacement field, pyvips does not have uniform sampling (yet)
    # instead, use a Gaussian and convert using Box-Muller inverse
    z1 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)
    z2 = pyvips.Image.gaussnoise(width, height, sigma=1.0, mean=0.0)

    # Compute box-muller inverse to get approximate uniform values
    dx = (-(z1 * z1 + z2 * z2) / 2).exp()
    dx = (2 * dx - 1).gaussblur(sigma) * alpha

    dy = (z1 / z2).atan()
    dy = (2 * dy - 1).gaussblur(sigma) * alpha

    grid = pyvips.Image.xyz(img.width, img.height)
    new_coords = grid + dx.bandjoin([dy])

    image = img.mapim(new_coords, interpolate=interpolation, background=background)

    return image


if __name__ == "__main__":
    # image = np.array(Image.open("../images/he.jpg"))
    # image = pyvips.Image.new_from_array(image)

    image = pyvips.Image.new_from_file(
        "/data/pathology/archives/breast/camelyon/CAMELYON16/images/normal_001.tif",
        level=2,
    ).flatten()

    print(image.width, image.height, image.bands)
    start = time.time()
    image = elastic_transform(image)

    image.write_to_file(
        "/data/pathology/projects/pathology-bigpicture-streamingclam/test_elastic.tif",
        pyramid=True,
        bigtiff=True,
        compression="jpeg",
        Q=90,
    )
    print(f"pyvips: {time.time() - start}s")
    # Image size: 24448 55293, camelyon16 normal 1
    # Mine: 340
    # Mine @ no box-muller: 170, 166
    # Hans @ grid scale 8: 260
    # Hans @ grid scale 16: 257
    # Hans @ grid scale 64: 257
