from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import albumentations as A
import pyvips
import matplotlib.pyplot as plt
from typing import Any


class ComposeV2(A.Compose):
    def _check_args(self, **kwargs) -> None:
        checked_single = ["image", "mask"]
        checked_multi = ["masks"]
        check_bbox_param = ["bboxes"]
        # ["bboxes", "keypoints"] could be almost any type, no need to check them
        shapes = []
        for data_name, data in kwargs.items():
            internal_data_name = self.additional_targets.get(data_name, data_name)
            if internal_data_name in checked_single:
                if isinstance(data, np.ndarray):
                    shapes.append(data.shape[:2])
                elif isinstance(data, pyvips.Image):
                    shapes.append((data.width, data.height))
                else:
                    raise TypeError("{} must be numpy array type".format(data_name))
            if internal_data_name in checked_multi:
                if data is not None:
                    if not isinstance(data[0], np.ndarray):
                        shapes.append(data[0].shape[:2])
                    elif isinstance(data[0], pyvips.Image):
                        shapes.append((data.width, data.height))
                    else:
                        raise TypeError(
                            "{} must be list of numpy arrays".format(data_name)
                        )
            if (
                internal_data_name in check_bbox_param
                and self.processors.get("bboxes") is None
            ):
                raise ValueError(
                    "bbox_params must be specified for bbox transformations"
                )

        if self.is_check_shapes and shapes and shapes.count(shapes[0]) != len(shapes):
            raise ValueError(
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency)."
            )


class RandomA(ImageOnlyTransform):
    """
    RamdomA transformation
    """

    def __init__(self, safe_db_lists=[], prob=0.5) -> None:
        super(RandomA, self).__init__()
        self.safe_db_lists = safe_db_lists
        self.prob = prob

    def apply(self, img: pyvips.Image, copy=True, **params):
        img = img.rot90()
        # some of your logic here
        return img

    def update_params(self, params: dict[str, Any], **kwargs) -> dict[str, Any]:
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        if hasattr(self, "mask_fill_value"):
            params["mask_fill_value"] = self.mask_fill_value
        params.update({"cols": kwargs["image"].width, "rows": kwargs["image"].height})
        return params


if __name__ == "__main__":
    image_1 = np.random.random(size=(256, 256, 3))
    image_2 = np.ones((256, 256, 3))
    image = np.append(image_1, image_2, axis=0)
    plt.imshow(image)
    plt.show()

    image = pyvips.Image.new_from_array(image)
    transforms = ComposeV2(
        transforms=[
            RandomA(prob=1.0),
        ]
    )
    # transforms._disable_check_args()

    image_dict = transforms(image=image)
    new_image = image_dict["image"]

    new_image = new_image.numpy()
    plt.imshow(new_image)
    plt.show()
