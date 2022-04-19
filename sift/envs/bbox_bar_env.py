import gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sift.envs import BBoxEnv

# NOTE:
# Faster-RCNN Architecture
# 1. image -> backbone -> feature map (H x W)
# 2. assign a set of anchors to each pixel in the feature map; anchors = map[i, j] = 3 scales x 3 ratios = 9 anchors for map[i, j]
# 3. apply 3x3, 512 unit convolution to feature map
# 4. apply two branch 1x1 convolution layers:
#   a. 18 unit layer for binary classification (9 proporsals x 2 classes) of a point in a feature map containing an object or not
#   b. 36 unit layer for bounding box regression (THIS IS PROBABLY WHERE SIFT WOULD PLUG IN)


class BAREnv(BBoxEnv):
    def __init__(
        self,
        backbone: nn.Module,
        dataloader: DataLoader,
        episode_length: int,
        beta: float,
        c: float,
    ) -> None:
        super().__init__(backbone, dataloader, episode_length, beta)

        self.c1 = c
        self.c2 = c / 2

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,))
        self.action_space.n = self.action_space.shape[0]

        self.history_vector = torch.zeros((self.episode_length, self.action_space.n))

        feature_dims = backbone.model.num_features
        history_dims = self.history_vector.view(-1, 1).shape[0]
        observation_dims = feature_dims + history_dims
        self.observation_space = gym.spaces.Box(
            low=-torch.inf,
            high=torch.inf,
            shape=(observation_dims,)
        )

    def _perform_action(self, action):
        image_height = self.ct["image"].shape[0]
        image_width = self.ct["image"].shape[1]
        region_height = self.ct["region_image"].shape[0]
        region_width = self.ct["region_image"].shape[1]
        ct_box = self.ct["instances"].ct_boxes[self.idx]

        if action == 0:  # up
            ct_box[0] = ct_box[0] - self.c1 * region_height
            ct_box[2] = ct_box[2] - self.c1 * region_height
        elif action == 1:  # down
            ct_box[0] = ct_box[0] + self.c1 * region_height
            ct_box[2] = ct_box[2] + self.c1 * region_height
        elif action == 2:  # left
            ct_box[1] = ct_box[1] - self.c1 * region_width
            ct_box[3] = ct_box[3] - self.c1 * region_width
        elif action == 3:  # right
            ct_box[1] = ct_box[1] + self.c1 * region_width
            ct_box[3] = ct_box[3] + self.c1 * region_width
        elif action == 4:  # wider
            ct_box[1] = ct_box[1] - self.c2 * region_width
            ct_box[3] = ct_box[3] + self.c2 * region_width
        elif action == 5:  # taller
            ct_box[0] = ct_box[0] - self.c2 * region_height
            ct_box[2] = ct_box[2] + self.c2 * region_height
        elif action == 6:  # fatter
            ct_box[0] = ct_box[0] + self.c2 * region_height
            ct_box[2] = ct_box[2] - self.c2 * region_height
        elif action == 7:  # thinner
            ct_box[1] = ct_box[1] + self.c2 * region_width
            ct_box[3] = ct_box[3] - self.c2 * region_width

        # boundaries in case bbox exceeds image size
        ct_box[0] = max(ct_box[0], 0)
        ct_box[1] = max(ct_box[1], 0)
        ct_box[2] = min(ct_box[2], image_width)
        ct_box[3] = min(ct_box[3], image_height)

        self.ct["instances"].ct_boxes[self.idx] = ct_box

    def _update_reward(self) -> torch.Tensor:
        iou_next = self._compute_iou()
        if self._check_end_state():
            if iou_next > self.beta:
                return 6 + 4 * ((self.episode_length - self.t) / self.episode_length)
            else:
                return -3
        else:
            if iou_next > self.iou_last:
                self.iou_last = iou_next
                return 1
            else:
                self.iou_last = iou_next
                return -3

    def _update_history_vector(self, action: int) -> None:
        action_vector = torch.zeros((1, self.action_space.n))
        action_vector[0, action] = 1
        self.history_vector[self.t] = action_vector
        self.t += 1
