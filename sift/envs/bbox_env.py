from typing import Optional, Union, Tuple, Dict

import gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
from detectron2.structures import Instances, Boxes, pairwise_iou
import matplotlib.pyplot as plt


class BBoxEnv(gym.Env):
    def __init__(
        self,
        backbone: nn.Module,
        dataloader: DataLoader,
        episode_length: int,
        beta: float
    ) -> None:
        self.backbone = backbone
        self.dataloader = dataloader
        self.episode_length = episode_length
        self.idx = 0
        self.t = 0
        self.beta = beta
        self.iou_last = 0.0
        self.input_shape = (224, 224)  # TODO: derive this from backbone
        self.backbone_transform = T.Resize(
            size=self.input_shape,
            interpolation=T.InterpolationMode.BILINEAR
        )

        self._load_next_sample()

    def step(
        self,
        action: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict]:
        self._perform_action(action)
        self._update_history_vector(action)
        state = self._update_state()
        reward = self._update_reward()

        done = self._check_end_state()

        info = {}

        return (state, reward, done, info)

    def reset(self) -> torch.Tensor:
        if self.idx == self.gt["instances"].__len__() - 1:
            self._load_next_sample()
            self.idx = 0
        else:
            self.idx += 1
            self._update_region()

        self._reset_history_vector()
        self.t = 0
        self.iou_last = 0.0
        state = self._update_state()

        return state

    def render(
        self,
        image,
        classes: Optional[list] = None,
        gt_boxes: Optional[list] = None,
        at_boxes: Optional[list] = None,
    ) -> None:
        gt_image = PILToTensor()(image).numpy()
        gt_image = gt_image.transpose((1, 2, 0))
        box_targets = []
        colors = []

        if gt_boxes is not None:
            transformed = self.transforms(
                image=gt_image,
                bboxes=gt_boxes,
                class_labels=classes
            )
            image = torch.tensor(transformed["image"]).permute(2, 0, 1)
            boxes = torch.tensor(transformed["bboxes"])
            box_targets.append(boxes)
            colors.append("red")
        if at_boxes is not None:
            transformed = self.transforms(
                image=gt_image,
                bboxes=at_boxes,
                class_labels=classes
            )
            image = torch.tensor(transformed["image"]).permute(2, 0, 1)
            boxes = torch.tensor(transformed["bboxes"])
            box_targets.append(boxes)
            colors.append("blue")

        for boxes, color in zip(box_targets, colors):
            image = draw_bounding_boxes(image, boxes, colors=color)

        plt.imshow(image.permute(1, 2, 0))
        plt.show()

    def _load_next_sample(self) -> None:
        image, gt_boxes, ct_boxes, labels = next(iter(self.dataloader))

        gt_annotations = {"gt_boxes": gt_boxes, "gt_classes": labels}
        gt_instance = Instances(image.shape[-2:], **gt_annotations)
        ct_annotations = {"ct_boxes": ct_boxes, "ct_classes": labels}
        ct_instance = Instances(image.shape[-2:], **ct_annotations)

        self.gt = {"image": image, "region_image": image, "instances": gt_instance}
        self.ct = {"image": image, "region_image": image, "instances": ct_instance}

    def _perform_action(self, action):
        raise NotImplementedError

    def _check_end_state(self):
        iou = self._compute_iou()
        return (iou > self.beta) or (self.t == self.episode_length)

    def _compute_iou(self):
        gt_box = Boxes(self.gt["instances"].gt_boxes[self.idx].unsqueeze(0))
        ct_box = Boxes(self.ct["instances"].ct_boxes[self.idx].unsqueeze(0))
        return pairwise_iou(gt_box, ct_box)

    def _reset_history_vector(self) -> None:
        self.history_vector = torch.zeros((self.episode_length, self.action_space.n))

    def _update_state(self) -> torch.Tensor:
        x = self.backbone_transform(self.ct["region_image"])
        features = self.backbone(x.unsqueeze(0))
        return torch.hstack((features, self.history_vector.view(1, -1)))

    def _update_reward(self) -> torch.Tensor:
        raise NotImplementedError

    def _update_region(self) -> None:
        gt_box = self.gt["instances"].gt_boxes[self.idx].int()
        ct_box = self.ct["instances"].ct_boxes[self.idx].int()
        self.gt["region_image"] = self.gt["image"][:, gt_box[0]:gt_box[2], gt_box[1]:gt_box[3]]
        self.ct["region_image"] = self.ct["image"][:, ct_box[0]:ct_box[2], ct_box[1]:ct_box[3]]

    def _update_history_vector(self, action: Union[int,torch.Tensor]) -> None:
        raise NotImplementedError
