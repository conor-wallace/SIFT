from typing import Union
import logging

import gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gym_bbox.envs import BBoxEnv


class SIFTEnv(BBoxEnv):
    """Search and Inspection For Trustworthy (SIFT) data

    Args:
        gym (_type_): _description_
    """
    def __init__(
        self,
        backbone: nn.Module,
        dataloader: DataLoader,
        episode_length: int,
        beta: float
    ) -> None:
        super().__init__(backbone, dataloader, episode_length, beta)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
        self.action_space.n = self.action_space.shape[0]

        self.history_vector = torch.zeros((self.episode_length, self.action_space.n))

        feature_dims = self.backbone.feature_dims
        history_dims = self.history_vector.view(-1, 1).shape[0]
        observation_dims = feature_dims + history_dims
        self.observation_space = gym.spaces.Box(
            low=-torch.inf,
            high=torch.inf,
            shape=(observation_dims,)
        )

    def _perform_action(self, action: Union[int, torch.Tensor]):
        image_height = self.ct["image"].shape[0]
        image_width = self.ct["image"].shape[1]
        ct_box = self.ct["instances"].ct_boxes[self.idx]
        ct_box = ct_box + ct_box * action

        # boundaries in case bbox exceeds image size
        ct_box[0] = max(ct_box[0], 0)
        ct_box[1] = max(ct_box[1], 0)
        ct_box[2] = min(ct_box[2], image_width)
        ct_box[3] = min(ct_box[3], image_height)

        self.ct["instances"].ct_boxes[self.idx] = ct_box

    def _update_reward(self) -> torch.Tensor:
        reward = self._compute_iou()
        logging.deubg(reward)
        return reward

    def _update_history_vector(self, action: Union[int, torch.Tensor]) -> None:
        self.history_vector[self.t] = action
        self.t += 1
