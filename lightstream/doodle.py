import albumentations as A
import numpy as np
import pyvips
import matplotlib.pyplot as plt
from PIL import Image
from lightstream.transforms import Compose, HorizontalFlip, VerticalFlip


if __name__ == "__main__":
    image = np.array(Image.open("astronaut.jpg"))
    plt.imshow(image)
    plt.title("original image")
    plt.show()

    print("image type", type(image))
    pyvips_image = pyvips.Image.new_from_array(image)
    transforms = A.Compose(
        transforms=[
            A.VerticalFlip(p=1.0),
        ]
    )
    # transforms._disable_check_args()

    image_dict = transforms(image=image)
    new_image = image_dict["image"]
    plt.imshow(new_image)
    plt.title("image after transform")
    plt.show()

    ## Pyvips equivalent

    transforms_pyvips = Compose(
        transforms=[
            VerticalFlip(p=1.0),
        ]
    )

    image_dict = transforms_pyvips(image=pyvips_image)
    new_image = image_dict["image"]
    new_image = new_image.numpy()
    plt.imshow(new_image)
    plt.title("pyvips image")
    plt.show()
