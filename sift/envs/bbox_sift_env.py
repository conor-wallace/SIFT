from typing import Union, Optional
import copy

import gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from detectron2.structures import Boxes

from sift.envs import BBoxEnv


class SIFTEnv(BBoxEnv):
    """Search and Inspection For Trustworthy (SIFT) data

    Args:
        gym (_type_): _description_
    """
    def __init__(
        self,
        backbone: nn.Module,
        dataloader: DataLoader,
        episode_length: Optional[int] = 10,
        num_actions: Optional[int] = 4,
        row: Optional[float] = 10.0,
        beta: Optional[float] = 0.9
    ) -> None:
        super().__init__(backbone, dataloader, episode_length, beta)

        self.row = row
        self.action_space = gym.spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(num_actions,)
        )
        self.action_space.n = self.action_space.shape[0]

        self.history_vector = torch.zeros(
            (self.episode_length, self.action_space.n)
        )

        feature_dims = backbone.model.num_features
        history_dims = self.history_vector.view(-1, 1).shape[0]
        observation_dims = feature_dims + history_dims
        self.observation_space = gym.spaces.Box(
            low=-torch.inf,
            high=torch.inf,
            shape=(observation_dims,)
        )

    def _perform_action(self, action: torch.Tensor):
        image_height = self.ct["image"].shape[1]
        image_width = self.ct["image"].shape[2]
        ct_box = copy.deepcopy(self.ct["instances"].ct_boxes[self.idx])
        ct_box = ct_box.float().cpu().numpy()

        # regression coefficients
        t_xmin, t_ymin, t_xmax, t_ymax = action
        ct_w = ct_box[2] - ct_box[0]
        ct_h = ct_box[3] - ct_box[1]

        ct_box[0] = t_xmin * ct_w + ct_box[0]
        ct_box[1] = t_ymin * ct_h + ct_box[1]
        ct_box[2] = t_xmax * ct_w + ct_box[2]
        ct_box[3] = t_ymax * ct_h + ct_box[3]

        # boundaries in case bbox exceeds image size
        ct_box[0] = round(max(ct_box[0], 0))
        ct_box[1] = round(max(ct_box[1], 0))
        ct_box[2] = round(min(ct_box[2], image_width))
        ct_box[3] = round(min(ct_box[3], image_height))

        self.ct["instances"].ct_boxes[self.idx] = torch.tensor(ct_box)

    def _compute_distance(self):
        gt_box = Boxes(self.gt["instances"].gt_boxes[self.idx].unsqueeze(0))
        ct_box = Boxes(self.ct["instances"].ct_boxes[self.idx].unsqueeze(0))
        gt_center = gt_box.get_centers()
        ct_center = ct_box.get_centers()
        return torch.mean((gt_center - ct_center) ** 2)

    def _compute_metrics(self):
        iou = self._compute_iou()
        distance = self._compute_distance()
        return iou, distance

    def _update_reward(self, verbose: bool = False) -> torch.Tensor:
        iou, distance = self._compute_metrics()
        r_iou = self.row * iou
        r_distance = torch.log(1.0 / distance)
        t_penalty = ((self.episode_length - self.t) / self.episode_length) + 1
        reward = (r_iou + r_distance) * t_penalty
        if verbose:
            print("R IoU: ", r_iou)
            print("R Distance: ", r_distance)
            print("T Penalty: ", t_penalty)
            print("Reward: ", reward)
            print("T: ", self.t)
            print("IoU Next, IoU Last: [{}, {}]".format(iou, self.iou_last))

        self.iou_last = iou

        return reward

        # if self._check_end_state():
        #     if iou > self.beta:
        #         penalty = (self.episode_length - self.t) / self.episode_length
        #         return torch.tensor(6 + 4 * penalty, dtype=torch.float32)
        #     else:
        #         return torch.tensor(-3, dtype=torch.float32)
        # else:
        #     if iou > self.iou_last:
        #         self.iou_last = iou
        #         return torch.tensor(1, dtype=torch.float32)
        #     else:
        #         self.iou_last = iou
        #         reward = iou 
        #         return torch.tensor(reward, dtype=torch.float32)

    def _update_history_vector(self, action: Union[int, torch.Tensor]) -> None:
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        self.history_vector[self.t] = action
        self.t += 1
