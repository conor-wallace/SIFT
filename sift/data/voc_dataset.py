from typing import Tuple, Dict, Any, Optional, Callable
import copy
import os
from attr import attributes

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor
from torchvision.datasets import VOCDetection as _VOCDetection
import pytorch_lightning as pl
from PIL import Image
from xml.etree.ElementTree import parse as ET_parse
from detectron2.structures import Boxes, Instances

VOC_CLASSES = {
    "background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}


def detectron2_collate_fn(batch: Dict[str, Any]) -> Instances:
    """Collate samples into a list of dictionaries containing images and
    anotation instances for training a model in detectron2 format.

    NOTE: detectron2 expects target boxes to have name `gt_boxes` for
    computing regression loss, so we swap the names of the two sets of boxes
    such that:
        {detectron2} gt_boxes = {here} ct_boxes
        {detectron2} original_gt_boxes = {here} gt_boxes

    Args:
        batch (Dict[str, Any]): un-collated batch of samples

    Returns:
        Instances: detectron2 formatted batch
    """
    images = torch.stack([sample["image"] for sample in batch])
    gt_boxes = [sample["gt_boxes"] for sample in batch]
    ct_boxes = [sample["ct_boxes"] for sample in batch]
    labels = [sample["labels"] for sample in batch]

    batched_inputs = []
    for i in range(images.shape[0]):
        image = images[i]
        annots = {
            "gt_boxes": Boxes(ct_boxes[i]),
            "original_gt_boxes": Boxes(gt_boxes[i]),
            "gt_classes": labels[i]
        }
        instances = Instances(image.shape[1:], **annots)
        sample = dict(image=image, instances=instances)
        batched_inputs.append(sample)

    return batched_inputs


class VOCCorruptionDataset(_VOCDetection):
    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        transforms: Optional[Callable] = None,
        corruption_transforms: Optional[Callable] = None
    ):
        if os.path.exists(root):
            download = False
        else:
            download = True
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download
        )
        self.transforms = transforms
        self.corruption_transforms = corruption_transforms

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dict of the XML tree.
        """
        image = self._load_image(self.images[idx])
        targets = self.parse_voc_xml(ET_parse(self.annotations[idx]).getroot())
        boxes, labels = self._load_targets(targets)

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        boxes_c = copy.deepcopy(boxes)
        if self.corruption_transforms is not None:
            for idx in range(len(boxes)):
                corrupted = self.corruption_transforms(
                    image=image,
                    bboxes=boxes,
                    class_labels=labels
                )
                boxes_c[idx] = corrupted["bboxes"][idx]

        image = torch.tensor(image).permute(2, 0, 1)
        boxes = torch.tensor(boxes)
        boxes_c = torch.tensor(boxes_c)
        labels = torch.tensor(labels)

        return dict(image=image, gt_boxes=boxes, ct_boxes=boxes_c, labels=labels)

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = PILToTensor()(image)
        image = image.permute((1, 2, 0))
        return image.numpy()

    def _load_targets(
        self,
        targets: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes = self._load_boxes(targets)
        labels = self._load_labels(targets)
        return boxes, labels

    def _load_boxes(self, targets: Dict) -> torch.Tensor:
        boxes = np.array([
                [
                    int(box['bndbox']['xmin']),
                    int(box['bndbox']['ymin']),
                    int(box['bndbox']['xmax']),
                    int(box['bndbox']['ymax'])
                ]
                for box in targets['annotation']['object']
            ]
        )
        return boxes

    def _load_labels(self, targets: Dict) -> torch.Tensor:
        return np.array(
            [VOC_CLASSES[x['name']] for x in targets['annotation']['object']]
        )


class VOCCorruptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        year: str = "2012",
        batch_size: int = 16,
        transforms: Optional[Callable] = None,
        corruption_transforms: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.root = root
        self.year = year
        self.batch_size = batch_size
        self.transforms = transforms
        self.corruption_transforms = corruption_transforms

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_dataset = VOCCorruptionDataset(
                root=self.root,
                year=self.year,
                image_set="train",
                transforms=self.transforms,
                corruption_transforms=self.corruption_transforms
            )

    def train_dataloader(self) -> DataLoader:
        print("BATCH_SIZE = ", self.batch_size)
        print("DATASET LEN = ", len(self.train_dataset))
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=detectron2_collate_fn,
        )
