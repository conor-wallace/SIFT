from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from pl_bolts.models.rl.common.agents import Agent


class DDPGAgent(Agent):
    """DDPG based agent that returns an action based on the networks policy."""
    
    def __init__(
        self,
        net: nn.Module,
        n_actions: int,
        action_high: float = 1.0,
        action_low: float = -1.0,
        eps: float = 0.1
    ) -> Agent:
        self.net = net
        self.n_actions = n_actions
        self.action_high = action_high
        self.action_low = action_low
        self.eps = eps

    def __call__(self, states: torch.Tensor, device: str) -> List[float]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy perturbed by white Gaussian noise
        """
        if not isinstance(states, Tensor):
            states = torch.tensor(states, device=device)

        mu = self.net(states)
        noise = self.eps * torch.randn(self.n_actions)
        actions = (mu + noise).clamp(self.action_low, self.action_high)

        return [actions.detach().cpu().numpy()]

    def get_action(self, states: torch.Tensor, device: str) -> List[float]:
        """Takes in the current state and returns the deterministic action from the agent's policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        if not isinstance(states, Tensor):
            states = torch.tensor(states, device=device)

        mu = self.net(states)
        actions = mu.detach().cpu().numpy()

        return [actions]