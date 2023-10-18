import albumentations as A
import numpy as np
import pyvips
import matplotlib.pyplot as plt
from PIL import Image
from lightstream.transforms import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    HEDShift,
    ElasticTransform,
    RandomRotate90,
)
from lightstream.transforms import color_aug_hed

import random

# random.seed(0)
# np.random.seed(0)


if __name__ == "__main__":
    image = np.array(Image.open("../images/he.jpg"))
    pyvips_image = pyvips.Image.new_from_array(image)
    print(pyvips_image.width, pyvips_image.height, pyvips_image.bands)
    plt.imshow(pyvips_image.numpy())
    plt.show()

    mask = (pyvips_image >= 250).bandand()
    plt.imshow(mask.numpy())
    plt.show()

    transforms = Compose([RandomRotate90(p=1.0)])
    sample = {'image': pyvips_image, 'mask': mask}
    new_image = transforms(**sample)

    plt.imshow(new_image["image"].numpy())
    plt.show()

    plt.imshow(new_image["mask"].numpy())
    plt.show()
