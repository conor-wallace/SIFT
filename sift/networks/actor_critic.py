from typing import Tuple

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_shape: Tuple[int], n_actions: int, hidden_size: int = 128):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return torch.tanh(self.net(x.float()).squeeze())


class Critic(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_shape: Tuple[int], n_actions: int, hidden_size: int = 128):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0] + n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=-1)
        return self.net(x.float()).squeeze()
