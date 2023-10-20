import pyvips

__all__ = ["random_crop"]


def random_crop(
    img: pyvips.Image, crop_height: int, crop_width: int, h_start: float, w_start: float
):
    width, height = img.width, img.height
    cx = max(0, int((width - crop_width + 1) * w_start))
    cy = max(0, int((height - crop_height + 1) * h_start))

    image = img.crop(cx, cy, min(img.width, crop_width), min(img.height, crop_height))

    return image
