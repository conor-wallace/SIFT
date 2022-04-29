from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from sift.data.buffer import Experience
from sift.agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """DQN Agent class handeling the interaction with the environment."""

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an
        epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            action = int(np.max(action))
        else:
            state = self.state.to(device)
            q_values = net(state)
            _, action = torch.max(q_values, dim=-1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu"
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            device: current device
            epsilon: value to determine likelihood of taking a random action

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done
