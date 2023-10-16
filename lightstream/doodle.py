import albumentations as A
import numpy as np
import pyvips
import matplotlib.pyplot as plt
from PIL import Image
from lightstream.transforms import Compose, HorizontalFlip, VerticalFlip, HEDShift
import os

if __name__ == "__main__":
    image = np.array(Image.open("../images/he.jpg"))
    transforms = Compose(
        [
            HEDShift(p=1.0)
        ])

    plt.imshow(image)
    plt.show()
    plt.imshow(transforms(image=pyvips.Image.new_from_array(image))['image'])
    plt.show()

    print((pyvips.Image.new_from_array(image) == 255).bandand())