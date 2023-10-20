import pyvips

__all__ = ["to_dtype", "normalize"]


def normalize(img: pyvips.Image, mean: list, std: list):
    img = (img - mean) / std
    return img



def to_dtype(img: pyvips.Image, dtype: str, scale=True):
    """

    Parameters
    ----------
    img : Pyvips.Image
        The input image
    dtype: str
        The dtype to convert to, can be one of uchar (uint8), float (float32) or double (float64)
    scale
        Whether to scale the input if converting from/to uchar

    Returns
    -------
    img: Pyvips.Image
        The dtype recasted image object
    """


    current_dtype = img.format

    if dtype not in ('uchar', 'float', 'double'):
        raise TypeError(f" dtype must be one of 'uchar', 'float32' or 'double', found {dtype}")

    if current_dtype == dtype:
        return img

    if current_dtype in ("float", "double") and dtype in ("float", "double"):
        return img.cast(dtype)

    if current_dtype in ("float", "double") and dtype in "uchar":
        img = img * 255.0 if scale else img
        img = img.cast(dtype)

    if current_dtype in "uchar" and dtype in ("float", "double"):
        img = img.cast(dtype)
        img = img / 255.0

    return img