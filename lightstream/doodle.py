import pyvips
import time

# Time transform resize from pyvips on uin8 data

gauss_image = pyvips.Image.gaussmat(5.0, 0.1, separable=False, precision="float")
print(gauss_image.bands, gauss_image.height, gauss_image.width)
print(gauss_image.numpy())