from examples.data.dataset import StreamingClassificationDataset
from pathlib import Path
from lightstream import transforms as T

import pyvips
import torch


class StreamingSurvivalDataset(StreamingClassificationDataset):
    def get_img_path(self, idx):
        img_fname = self.classification_frame.iloc[idx, 0]
        label = self.classification_frame.iloc[idx, 1]
        follow_up = self.classification_frame.iloc[idx, 2]

        img_path = self.img_dir / Path(img_fname).with_suffix(self.filetype)

        if self.mask_dir:
            mask_path = self.mask_dir / Path(img_fname + self.mask_suffix).with_suffix(self.filetype)
            return [img_path, mask_path], label, follow_up

        return [img_path], label, follow_up

    def check_csv(self):
        """Check if entries in csv file exist"""
        included = (
            {"images": [], "masks": [], "labels": [], "follow_up": []}
            if self.mask_dir
            else {"images": [], "labels": [], "follow_up": []}
        )

        for i in range(len(self)):
            images, label, follow_up = self.get_img_path(i)  #

            # Files can be just images, but also image, mask
            for file in images:
                if not file.exists():
                    print(f"WARNING {file} not found, excluded both image and mask (if present)!")
                    continue

            included["images"].append(images[0])
            included["labels"].append(label)
            included["follow_up"].append(follow_up)

            if self.mask_dir:
                included["masks"].append(images[1])

        self.data_paths = included

    def __getitem__(self, idx):
        img_fname = str(self.data_paths["images"][idx])
        label = int(self.data_paths["labels"][idx])
        follow_up = int(self.data_paths["follow_up"][idx])

        image = pyvips.Image.new_from_file(img_fname, page=self.read_level)
        sample = {"image": image}

        if self.mask_dir:
            mask_fname = str(self.data_paths["masks"][idx])
            mask = pyvips.Image.new_from_file(mask_fname)
            ratio = image.width / mask.width
            sample["mask"] = mask.resize(ratio, kernel="nearest")  # Resize mask to img size

        if self.transform:
            # print("applying transforms")
            sample = self.transform(**sample)

        # Output of transforms are uint8 images in the range [0,255]
        normalize = T.Compose(
            [
                T.PadIfNeeded(
                    pad_height_divisor=self.tile_delta,
                    pad_width_divisor=self.tile_delta,
                    min_height=None,
                    min_width=None,
                    value=[255, 255, 255],
                    mask_value=[0, 0, 0],
                ),
                T.ToDtype("float", scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        sample = normalize(**sample)

        # Masks don't need to be really large for tissues, so scale them back
        # TODO: Make transformation that operates on mask alone to resize
        if self.mask_dir:
            # Resize to streamingclam output stride, with max pool kernel
            sample["mask"] = sample["mask"].resize(1 / self.network_output_stride, kernel="nearest")

        to_tensor = T.Compose([T.ToTensor(transpose_mask=True)], is_check_shapes=False)
        sample = to_tensor(**sample)

        if self.mask_dir:
            sample["mask"] = sample["mask"] >= 1
            return sample["image"], sample["mask"], torch.tensor(label), torch.tensor(follow_up)

        return sample["image"], torch.tensor(label), torch.tensor(follow_up)
