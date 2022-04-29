from typing import Tuple, OrderedDict, List
import logging

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm

from sift.learners import BaseLightning
from sift.models import Actor, Critic
from sift.agents import DDPGAgent
from sift.data import ReplayBuffer, VOCCorruptionDataset
from sift.envs import SIFTEnv


class SIFTLightning(BaseLightning):
    def __init__(self, cfg: DictConfig) -> None:
        backbone = instantiate(cfg.model.backbone)
        dataloader = VOCCorruptionDataset(
            root=cfg.data.voc.root,
            year=cfg.data.voc.year,
            image_set=cfg.data.voc.image_set,
            download=cfg.data.voc.download,
            transforms=instantiate(cfg.data.transforms.voc_transforms),
            corruption_transforms=instantiate(
                cfg.data.transforms.corruption_transforms
            )
        )
        self.env = SIFTEnv(
            backbone=backbone,
            dataloader=dataloader,
            episode_length=cfg.data.buffer.episode_length,
            beta=cfg.model.sift.beta
        )
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.actor_net = Actor(obs_size, n_actions)
        self.target_actor_net = Actor(obs_size, n_actions)
        self.critic_net = Critic(obs_size, n_actions)
        self.target_critic_net = Critic(obs_size, n_actions)

        self.buffer = ReplayBuffer(cfg.data.buffer.replay_size)
        self.agent = DDPGAgent(self.env, self.buffer)
        super().__init__(cfg)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to
        initially fill up the replay buffer with experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in tqdm(
            range(steps),
            total=steps,
            desc="Populating experience buffer..."
        ):
            self.agent.play_step(self.actor_net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state x through the network and gets the actions as an output.

        Args:
            x: environment state

        Returns:
            continuous action
        """
        output = self.net(x)
        return output

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        states, actions, rewards, dones, next_states = batch

        q_values = self.critic_net(states, actions)
        next_actions = self.target_actor_net(next_states)
        next_q_values = self.target_critic_net(
            next_states,
            next_actions.detach()
        )
        expected_q_values = rewards + self.gamma * next_q_values

        critic_loss = nn.MSELoss()(q_values, expected_q_values)
        actor_loss = - self.critic(states, self.actor(states)).mean()
        return actor_loss, critic_loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> OrderedDict:
        """Carries out a single step through the environment to update the
        replay buffer. Then calculates loss based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(
            self.eps_end,
            self.eps_start - self.global_step + 1 / self.eps_last_frame,
        )

        # step through environment with agent
        reward, done = self.agent.play_step(self.actor_net, epsilon, device)
        self.episode_reward += reward

        actor_optimizer, critic_optimizer = self.optimizers()
        actor_loss, critic_loss = self.loss(batch)

        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        critic_optimizer.step()

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target networks
        if self.global_step % self.sync_rate == 0:
            self.target_actor_net.load_state_dict(
                self.actor_net.state_dict()
            )
            self.target_critic_net.load_state_dict(
                self.critic_net.state_dict()
            )

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_actor_loss": actor_loss,
            "train_critic_loss": critic_loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        if (self.global_step + 1) % 100 == 0:
            logging.debug("REWARD")
            logging.debug(self.total_reward)

        return OrderedDict(
            {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "log": log,
                "progress_bar": status
            }
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        actor_optimizer = Adam(self.net.parameters(), lr=self.lr)
        critic_optimizer = Adam(self.net.parameters(), lr=self.lr)
        return actor_optimizer, critic_optimizer
