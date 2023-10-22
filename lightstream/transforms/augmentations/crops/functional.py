import pyvips

__all__ = ["random_crop", "pad_or_crop"]


def random_crop(
    img: pyvips.Image, crop_height: int, crop_width: int, h_start: float, w_start: float
):
    width, height = img.width, img.height
    cx = max(0, int((width - crop_width + 1) * w_start))
    cy = max(0, int((height - crop_height + 1) * h_start))

    image = img.crop(cx, cy, min(img.width, crop_width), min(img.height, crop_height))

    return image


def pad_or_crop(
    img: pyvips.Image,
    crop_width: int,
    crop_height: int,
    background: list[int, int, int],
    direction: str = "centre",
) -> pyvips.Image:
    return img.gravity(direction, crop_width, crop_height, background=background)


def crop(img: pyvips.Image, x_min: int, y_min: int, x_max: int, y_max: int):
    height, width = img.height, img.width
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            "(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            "height = {height}, width = {width})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                height=height,
                width=width,
            )
        )

    print(f"img width and height during crop: {img.width}, {img.height}")
    print(f"xmin: {x_min} , xmax: {x_max}, ymin: {y_min}, ymax: {y_max}")

    width = x_max - x_min
    height = y_max - y_min

    print(f"args to crop: {x_min}, {y_min},{width}, {height}")
    return img.crop(x_min, y_min, width, height)
