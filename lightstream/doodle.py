import pyvips
import time
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from enum import Enum

# Time transform resize from pyvips on uin8 data

image = np.array(Image.open("../images/image_3.png"))
print(image[0:3, 0:3, 0])
pyvips_image = pyvips.Image.new_from_array(image).flatten()
print(pyvips_image.bands)
pyvips_image = pyvips_image.gravity(
    "centre",
    pyvips_image.width + 100,
    pyvips_image.height + 10,
    background=[255, 255, 255],
)

plt.imshow(pyvips_image.numpy())
plt.show()


class PositionType(Enum):
    CENTER = "centre"
    TOP_LEFT = "north-west"
    TOP_RIGHT = "north-east"
    BOTTOM_LEFT = "south-west"
    BOTTOM_RIGHT = "south-east"


print(PositionType["CENTER"].value)

z = "centre"
print(PositionType(z).value)
