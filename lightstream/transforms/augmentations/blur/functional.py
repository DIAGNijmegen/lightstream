import pyvips

__all__=["gaussian_blur"]
def gaussian_blur(
    img: pyvips.Image, sigma: float, min_amplitude: float
) -> pyvips.Image:
    return img.gaussblur(sigma, min_ampl=min_amplitude)
