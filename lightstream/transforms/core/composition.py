"""
Code is modified from https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
"""
import typing
import random
import pyvips

import numpy as np

from albumentations.core.composition import BaseCompose, get_always_apply
from albumentations.core.bbox_utils import BboxParams, BboxProcessor
from albumentations.core.keypoints_utils import KeypointParams, KeypointsProcessor
from albumentations.core.utils import get_shape

from .transforms_interface import BasicTransform

TransformType = typing.Union[BasicTransform, "BaseCompose"]
TransformsSeqType = typing.Sequence[TransformType]


class Compose(BaseCompose):
    """Compose transforms and handle all transformations regarding bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
        is_check_shapes (bool): If True shapes consistency of images/mask/masks would be checked on each call. If you
            would like to disable this check - pass False (do it only if you are sure in your data consistency).
    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: typing.Optional[typing.Union[dict, "BboxParams"]] = None,
        keypoint_params: typing.Optional[typing.Union[dict, "KeypointParams"]] = None,
        additional_targets: typing.Optional[typing.Dict[str, str]] = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
    ):
        super(Compose, self).__init__(transforms, p)

        self.processors: typing.Dict[
            str, typing.Union[BboxProcessor, KeypointsProcessor]
        ] = {}
        if bbox_params:
            if isinstance(bbox_params, dict):
                b_params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                b_params = bbox_params
            else:
                raise ValueError(
                    "unknown format of bbox_params, please use `dict` or `BboxParams`"
                )
            self.processors["bboxes"] = BboxProcessor(b_params, additional_targets)

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                k_params = KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, KeypointParams):
                k_params = keypoint_params
            else:
                raise ValueError(
                    "unknown format of keypoint_params, please use `dict` or `KeypointParams`"
                )
            self.processors["keypoints"] = KeypointsProcessor(
                k_params, additional_targets
            )

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

        self.is_check_args = True
        self._disable_check_args_for_transforms(self.transforms)

        self.is_check_shapes = is_check_shapes

    @staticmethod
    def _disable_check_args_for_transforms(transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                Compose._disable_check_args_for_transforms(transform.transforms)
            if isinstance(transform, Compose):
                transform._disable_check_args()

    def _disable_check_args(self) -> None:
        self.is_check_args = False

    def __call__(
        self, *args, force_apply: bool = False, **data
    ) -> typing.Dict[str, typing.Any]:
        if args:
            raise KeyError(
                "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            )
        if self.is_check_args:
            self._check_args(**data)
        assert isinstance(
            force_apply, (bool, int)
        ), "force_apply must have bool or int type"
        need_to_run = force_apply or random.random() < self.p
        for p in self.processors.values():
            p.ensure_data_valid(data)
        transforms = (
            self.transforms if need_to_run else get_always_apply(self.transforms)
        )

        check_each_transform = any(
            getattr(item.params, "check_each_transform", False)
            for item in self.processors.values()
        )

        for p in self.processors.values():
            p.preprocess(data)

        for idx, t in enumerate(transforms):
            data = t(**data)

            if check_each_transform:
                data = self._check_data_post_transform(data)
        data = Compose._make_targets_contiguous(
            data
        )  # ensure output targets are contiguous

        for p in self.processors.values():
            p.postprocess(data)

        return data

    def _check_data_post_transform(
        self, data: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        rows, cols = get_shape(data["image"])

        for p in self.processors.values():
            if not getattr(p.params, "check_each_transform", False):
                continue

            for data_name in p.data_fields:
                data[data_name] = p.filter(data[data_name], rows, cols)
        return data

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        dictionary = super(Compose, self)._to_dict()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict()
                if bbox_processor
                else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
                "additional_targets": self.additional_targets,
                "is_check_shapes": self.is_check_shapes,
            }
        )
        return dictionary

    def get_dict_with_id(self) -> typing.Dict[str, typing.Any]:
        dictionary = super().get_dict_with_id()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict()
                if bbox_processor
                else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
                "additional_targets": self.additional_targets,
                "params": None,
                "is_check_shapes": self.is_check_shapes,
            }
        )
        return dictionary

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

    @staticmethod
    def _make_targets_contiguous(
        data: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            result[key] = value
        return result
