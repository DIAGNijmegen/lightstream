from torchvision.transforms.v2._transform import Transform
from torchvision import tv_tensors
import PIL.Image
from torchvision.transforms import v2
from torchvision.transforms.v2._utils import check_type, has_any, is_pure_tensor
from torchvision.transforms.v2 import Compose
from typing import Any, List, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyvips
from torchvision.io import read_image
from pathlib import Path
from helpers import plot
from torchvision.transforms.v2 import functional as F


class PyvipsTransform(Transform):
    def __init__(self):
        super().__init__()
        torch, pil = self._transformed_types
        self._transformed_types = (torch, pil, pyvips.Image)

    def _needs_transform_list(self, flat_inputs: List[Any]) -> List[bool]:
        # Below is a heuristic on how to deal with simple tensor inputs:
        # 1. Simple tensors, i.e. tensors that are not a datapoint, are passed through if there is an explicit image
        #    (`datapoints.Image` or `PIL.Image.Image`) or video (`datapoints.Video`) in the sample.
        # 2. If there is no explicit image or video in the sample, only the first encountered simple tensor is
        #    transformed as image, while the rest is passed through. The order is defined by the returned `flat_inputs`
        #    of `tree_flatten`, which recurses depth-first through the input.
        #
        # This heuristic stems from two requirements:
        # 1. We need to keep BC for single input simple tensors and treat them as images.
        # 2. We don't want to treat all simple tensors as images, because some datasets like `CelebA` or `Widerface`
        #    return supplemental numerical data as tensors that cannot be transformed as images.
        #
        # The heuristic should work well for most people in practice. The only case where it doesn't is if someone
        # tries to transform multiple simple tensors at the same time, expecting them all to be treated as images.
        # However, this case wasn't supported by transforms v1 either, so there is no BC concern.

        needs_transform_list = []
        transform_pure_tensor = not has_any(
            flat_inputs,
            tv_tensors.Image,
            tv_tensors.Video,
            PIL.Image.Image,
            pyvips.Image,
        )

        for inpt in flat_inputs:
            needs_transform = True

            if not check_type(inpt, self._transformed_types):
                needs_transform = False
            elif is_pure_tensor(inpt):
                if transform_pure_tensor:
                    transform_pure_tensor = False
                else:
                    needs_transform = False
            needs_transform_list.append(needs_transform)
        return needs_transform_list


class RandomFlipPyvips(PyvipsTransform):
    def __init__(self):
        super().__init__()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        print("inpt", inpt)
        inpt = inpt.fliphor()
        return inpt


class RandomFlip(PyvipsTransform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.horizontal_flip, inpt)


if __name__ == "__main__":
    image_1 = np.random.random(size=(256, 256, 3))
    image_2 = np.ones((256, 256, 3))
    image = np.append(image_1, image_2, axis=0)
    mask = image[..., 0]

    img = read_image(str(Path("astronaut.jpg")))

    print(f"{type(img) = }, {img.dtype = }, {img.shape = }")
    transform = v2.RandomCrop(size=(224, 224))
    out = transform(img)

    plot([img, out])

    plt.show()

    boxes = tv_tensors.BoundingBoxes(
        [[15, 10, 370, 510], [275, 340, 510, 510], [130, 345, 210, 425]],
        format="XYXY",
        canvas_size=img.shape[-2:],
    )

    transforms = v2.Compose(
        [
            RandomFlip(),
        ]
    )
    out_img, out_boxes = transforms(img, boxes)
    print("output types", type(boxes), type(out_boxes))
    out_img = out_img.numpy()
    out_img = torch.from_numpy(out_img)
    plot([(img, boxes), (out_img, out_boxes)])
    plt.show()

    """

    plt.imshow(image)
    plt.show()

    plt.imshow(mask)
    plt.show()

    image = pyvips.Image.new_from_array(image)
    mask = pyvips.Image.new_from_array(mask)
    transforms = Compose(
        [
            RandomRotationPyvips(),
        ]
    )

    print(type(image))
    random_crap = [1, 2, 3]
    new_images = transforms(image, mask, random_crap)
    new_image, new_mask, stuff = new_images
    print(stuff)
    new_image = new_image.numpy()
    new_mask = new_mask.numpy()
    plt.imshow(new_image)
    plt.show()

    plt.imshow(new_mask)
    plt.show()
    """
