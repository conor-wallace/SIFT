from typing import Tuple, Dict, Any, Optional, Callable
import numpy as np
import torch
from torchvision.transforms import PILToTensor
from torchvision.datasets import VOCDetection as _VOCDetection
from PIL import Image
from xml.etree.ElementTree import parse as ET_parse
from detectron2.structures import Boxes


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


class VOCCorruptionDataset(_VOCDetection):
    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transforms: Optional[Callable] = None,
        corruption_transforms: Optional[Callable] = None
    ):
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

        if self.corruption_transforms is not None:
            corrupted = self.corruption_transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            boxes_c = corrupted["bboxes"]

        image = torch.tensor(image).permute(2, 0, 1)
        boxes = torch.tensor(boxes)
        boxes_c = torch.tensor(boxes_c)
        labels = torch.tensor(labels)

        return image, boxes, boxes_c, labels

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
