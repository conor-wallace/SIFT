from typing import Tuple

import gym
import torch
import torch.nn as nn

from sift.data import ReplayBuffer


class BaseAgent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, device: str) -> int:
        """Using the given network, decide what action to carry out.

        Args:
            net: Policy network {DQN, Actor, ...}
            device: current device

        Returns:
            action
        """
        raise NotImplementedError

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: Policy network {DQN, Actor, ...}
            device: current device

        Returns:
            reward, done
        """
        raise NotImplementedError
