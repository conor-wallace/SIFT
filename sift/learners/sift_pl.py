from typing import Tuple, Optional, Callable, OrderedDict, List
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm
from gym import Env

from pl_bolts.datamodules.experience_source import Experience, ExperienceSourceDataset
from pl_bolts.models.rl.common.memory import MultiStepBuffer

from sift.learners import BaseLightning
from sift.networks import Actor, Critic
from sift.agents import DDPGAgent
from sift.data import ReplayBuffer, VOCCorruptionDataset
from sift.envs import SIFTEnv
from sift.networks.backbones import ResNet50


class SIFTLightning(pl.LightningModule):
    def __init__(
        self,
        seed: Optional[int] = 123,
        backbone: Optional[Callable] = ResNet50,
        voc_transforms: Optional[Callable] = nn.Identity,
        corruption_transforms: Optional[Callable] = nn.Identity,
        root: Optional[str] = None,
        year: Optional[str] = "2012",
        image_set: Optional[str] = "train",
        download: Optional[bool] = False,
        beta: Optional[float] = 0.9,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 150000,
        sync_rate: int = 1000,
        gamma: float = 0.99,
        tau: float = 1e-3,
        hidden_size: int = 128,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        batch_size: int = 32,
        num_workers: int = 0,
        replay_size: int = 100000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -21,
        batches_per_epoch: int = 1000,
        n_steps: int = 1,
    ) -> None:
        super().__init__()

        # Environment
        self.env = self.make_environment(
            seed=seed,
            backbone=backbone,
            voc_transforms=voc_transforms,
            corruption_transforms=corruption_transforms,
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            beta=beta
        )

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.shape[0]

        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.build_networks(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            hidden_size=hidden_size
        )

        self.agent = DDPGAgent(self.actor)

        # Hyperparameters
        self.save_hyperparameters()

        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0

        # Average Rewards
        self.avg_reward_len = avg_reward_len

        for _ in range(avg_reward_len):
            self.total_rewards.append(torch.tensor(min_episode_reward, device=self.device))

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

        self.state = self.env.reset()

        self.automatic_optimization = False

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

    def build_networks(
        self,
        obs_shape: Tuple[int],
        n_actions: int,
        hidden_size: Optional[int] = 128
    ) -> None:
        """Initializes the DDPG actor and critic networks (with targets)"""
        self.actor = Actor(
            obs_shape=obs_shape,
            n_actions=n_actions,
            hidden_size=hidden_size
        )
        self.actor_target = Actor(
            obs_shape=obs_shape,
            n_actions=n_actions,
            hidden_size=hidden_size
        )
        self.critic = Critic(
            obs_shape=obs_shape,
            n_actions=n_actions,
            hidden_size=hidden_size
        )
        self.critic_target = Critic(
            obs_shape=obs_shape,
            n_actions=n_actions,
            hidden_size=hidden_size
        )

    def soft_update_target(self, net, target_net):
        """Update the weights in target network using a weighted sum.
        w_target := (1-a) * w_target + a * w
        Args:
            net: network
            target_net: target network
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                (1.0 - self.tau) * target_param.data + self.tau * param
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state x through the network and gets the actions as an output.

        Args:
            x: environment state

        Returns:
            continuous action
        """
        output = self.net(x)
        return output

    def train_batch(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.

        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1
            action = self.agent(self.state, self.device)

            next_state, reward, done, _ = self.env.step(action[0])

            episode_reward += reward
            episode_steps += 1

            exp = Experience(
                state=self.state, action=action[0], reward=reward, done=done, new_state=next_state
            )

            self.buffer.append(exp)
            self.state = next_state

            if done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(self.hparams.batch_size)

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.hparams.batches_per_epoch == 0:
                break

    def loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        states, actions, rewards, dones, next_states = batch

        q_values = self.critic_net(states, actions)
        next_actions = self.target_actor_net(next_states)
        next_q_values = self.target_critic_net(
            next_states,
            next_actions.detach()
        )
        expected_q_values = rewards + (1 - dones.int()) * self.gamma * next_q_values
        critic_loss = F.mse_loss(q_values, expected_q_values)

        actions = self.actor_net(states)
        actor_loss = -self.critic_net(states, actions).mean()

        return actor_loss, critic_loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> OrderedDict:
        """Carries out a single step through the environment to update the
        replay buffer. Then calculates loss based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
        """
        actor_optim, critic_optim = self.optimizers()
        actor_loss, critic_loss = self.loss(batch)

        actor_optim.zero_grad()
        self.manual_backward(actor_loss)
        actor_optim.step()

        critic_optim.zero_grad()
        self.manual_backward(critic_loss)
        critic_optim.step()

        # Soft update of target networks
        if self.global_step % self.sync_rate == 0:
            self.soft_update_target(self.critic, self.critic_target)
            self.soft_update_target(self.actor, self.actor_target)
        
        self.log_dict(
            {
                "total_reward": self.total_rewards[-1],
                "avg_reward": self.avg_rewards,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        actor_optimizer = Adam(self.actor.parameters(), lr=self.hparams.actor_lr)
        critic_optimizer = Adam(self.critic.parameters(), lr=self.hparams.critic_lr)
        return actor_optimizer, critic_optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.hparams.replay_size, self.hparams.n_steps)
        self.populate(self.hparams.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def make_environment(
        seed: Optional[int] = None,
        backbone: Optional[Callable] = ResNet50,
        voc_transforms: Optional[Callable] = nn.Identity,
        corruption_transforms: Optional[Callable] = nn.Identity,
        root: Optional[str] = None,
        year: Optional[str] = "2012",
        image_set: Optional[str] = "train",
        download: Optional[bool] = False,
        beta: Optional[float] = 0.9
    ) -> Env:
        """Initialise BAR environment.

        Args:
            seed: value to seed the environment RNG for reproducibility
        Returns:
            gym environment
        """
        dataloader = VOCCorruptionDataset(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transforms=voc_transforms,
            corruption_transforms=corruption_transforms
        )
        env = SIFTEnv(
            backbone=backbone,
            dataloader=dataloader,
            beta=beta
        )

        return env