from typing import Optional, Union, Tuple, Dict

import cv2
import gym
from numpy import float32
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
from detectron2.structures import Instances, Boxes, pairwise_iou
from detectron2.utils.visualizer import Visualizer
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
        action: Union[int, torch.Tensor],
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict]:
        if verbose:
            print("action: ", action)
        self._perform_action(action)
        self._update_history_vector(action)
        state = self._update_state()
        reward = self._update_reward(verbose=verbose)

        done = self._check_end_state()

        info = {}

        if verbose:
            self.render()

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
        self.iou_last = self._compute_iou()
        state = self._update_state()

        return state

    def render(self) -> None:
        vis = Visualizer(self.gt["image"].permute(1, 2, 0))
        image = vis.draw_box(tuple(self.gt["instances"].gt_boxes[self.idx]), edge_color="blue")
        image = vis.draw_box(tuple(self.ct["instances"].ct_boxes[self.idx]), edge_color="red")
        cv2.imshow('color image', image.get_image())
        cv2.waitKey(5)
        cv2.destroyAllWindows()

    def _load_next_sample(self) -> None:
        image, gt_boxes, ct_boxes, labels = next(iter(self.dataloader))

        gt_annotations = {"gt_boxes": gt_boxes, "gt_classes": labels}
        gt_instance = Instances(image.shape[-2:], **gt_annotations)
        ct_annotations = {"ct_boxes": ct_boxes, "ct_classes": labels}
        ct_instance = Instances(image.shape[-2:], **ct_annotations)

        self.gt = {
            "image": image,
            "region_image": image,
            "instances": gt_instance
        }
        self.ct = {
            "image": image,
            "region_image": image,
            "instances": ct_instance
        }

    def _perform_action(self, action):
        raise NotImplementedError

    def _check_end_state(self):
        iou = self._compute_iou()
        return torch.tensor(
            (iou > self.beta) or (self.t == self.episode_length),
            dtype=torch.uint8
        )

    def _compute_iou(self):
        gt_box = Boxes(self.gt["instances"].gt_boxes[self.idx].unsqueeze(0))
        ct_box = Boxes(self.ct["instances"].ct_boxes[self.idx].unsqueeze(0))
        return pairwise_iou(gt_box, ct_box).squeeze()

    def _reset_history_vector(self) -> None:
        self.history_vector = torch.zeros(
            (self.episode_length, self.action_space.n)
        )

    def _update_state(self) -> torch.Tensor:
        x = self.backbone_transform(self.ct["region_image"])
        features = self.backbone(x.unsqueeze(0))
        return torch.hstack((features, self.history_vector.view(1, -1))).squeeze(0)

    def _update_reward(self) -> torch.Tensor:
        raise NotImplementedError

    def _update_region(self) -> None:
        gt_box = self.gt["instances"].gt_boxes[self.idx].int()
        ct_box = self.ct["instances"].ct_boxes[self.idx].int()
        if ct_box[2] == ct_box[0]:
            ct_box[2] += 1
        if ct_box[3] == ct_box[1]:
            ct_box[3] += 1
        self.gt["region_image"] =\
            self.gt["image"][:, gt_box[0]:gt_box[2], gt_box[1]:gt_box[3]]
        self.ct["region_image"] =\
            self.ct["image"][:, ct_box[0]:ct_box[2], ct_box[1]:ct_box[3]]

    def _update_history_vector(self, action: Union[int, torch.Tensor]) -> None:
        raise NotImplementedError
