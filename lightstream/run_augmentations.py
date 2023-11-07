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
    ToDtype,
    Normalize,
    RandomCrop,
    GaussianBlur,
    CropOrPad,
    Rotate,
    HueSaturationValue,
    RandomBrightnessContrast,
    RandomGamma,
    GaussNoise,
    PadIfNeeded,
)
from lightstream.transforms import color_aug_hed

import random

# random.seed(0)
# np.random.seed(0)

import cv2

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    image = np.array(Image.open("../images/he.jpg"))

    plt.imshow(image)
    plt.show()

    pyvips_image = pyvips.Image.new_from_array(image)
    transforms = Compose([Rotate(), HEDShift(p=0.5), VerticalFlip(), HorizontalFlip()], is_check_shapes=True)

    sample = {"image": pyvips_image}
    new_image = transforms(**sample)

    plt.imshow(new_image["image"].numpy())
    plt.show()
