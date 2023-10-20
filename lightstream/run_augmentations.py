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
    ToTensor,
    toDtype,
    Normalize,
    RandomCrop,
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

    mask = (pyvips_image >= 250).bandand()

    plt.imshow(mask.numpy(), alpha=0.4)

    plt.show()
    transforms = Compose([RandomCrop(200, 200, p=1.00)], is_check_shapes=False)

    sample = {"image": pyvips_image, "mask": mask}
    new_image = transforms(**sample)

    out_image = new_image["image"]
    out_mask = new_image["mask"]

    print(out_image.format)
    print(out_mask.format)

    plt.imshow(out_image.numpy())
    plt.imshow(out_mask.numpy(), alpha=0.3)
    plt.show()
