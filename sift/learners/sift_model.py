"""SIFT: Search and Inspection for Trustworthy Image Annotations."""
from tabnanny import verbose
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, seed_everything
from torch import Tensor, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from tqdm import tqdm

from pl_bolts.datamodules.experience_source import Experience, ExperienceSourceDataset
from pl_bolts.models.rl.common.memory import MultiStepBuffer
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from sift.envs import SIFTEnv
from sift.agents import DDPGAgent
from sift.networks import Actor, Critic
from sift.data import VOCCorruptionDataset

if _GYM_AVAILABLE:
    from gym import Env
else:  # pragma: no cover
    warn_missing_pkg("gym")
    Env = object


class SIFT(LightningModule):
    def __init__(
        self,
        eps: float = 0.1,
        sync_rate: int = 1,
        gamma: float = 0.99,
        policy_learning_rate: float = 3e-4,
        q_learning_rate: float = 3e-4,
        tau: float = 5e-3,
        hidden_size: int = 500,
        batch_size: int = 128,
        replay_size: int = 1000000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -21,
        batches_per_epoch: int = 10000,
        n_val_episodes: int = 100,
        n_steps: int = 1,
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__()

        # Environment
        self.env = self.make_environment(**kwargs)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.shape[0]

        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.policy = None
        self.policy_target = None
        self.q = None
        self.q_target = None
        self.build_networks(hidden_size = hidden_size)

        self.agent = DDPGAgent(
            self.policy,
            n_actions=self.n_actions,
            action_high=self.env.action_space.high[0],
            action_low=self.env.action_space.low[0],
            eps=eps
        )

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

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))

        self.state = self.env.reset()

        self.automatic_optimization = False

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience."""
        if warm_start > 0:
            self.state = self.env.reset()

            for _ in tqdm(
                range(warm_start),
                total=warm_start,
                desc="Populating experience buffer."
            ):
                action = self.agent(self.state, self.device)
                next_state, reward, done, _ = self.env.step(action[0])
                exp = Experience(
                    state=self.state.numpy(),
                    action=action[0],
                    reward=reward,
                    done=done,
                    new_state=next_state.numpy()
                )
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state = self.env.reset()

    def run_n_episodes(self) -> Tensor:
        """Carries out N episodes of the environment with the current agent without exploration."""
        total_rewards = []

        for i in range(self.hparams.n_val_episodes):
            print("Validation Episode #{}".format(i))
            episode_state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.agent.get_action(episode_state, self.device)
                next_state, reward, done, _ = self.env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        yield torch.tensor(total_rewards, dtype=torch.float32)
        _ = self.env.reset()

    def generate_batch(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.
        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1
            action = self.agent(self.state, self.device)

            next_state, reward, is_done, _ = self.env.step(action[0])

            episode_reward += reward
            episode_steps += 1

            exp = Experience(
                state=self.state.numpy(),
                action=action[0],
                reward=reward,
                done=is_done,
                new_state=next_state.numpy()
            )

            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(self.hparams.batch_size)

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.hparams.batches_per_epoch == 0:
                break

    def build_networks(self, hidden_size: Optional[int] = 500) -> None:
        """Initializes the DDPG actor and critic networks (with targets)"""
        self.policy = Actor(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            hidden_size=hidden_size
        )
        self.policy_target = Actor(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            hidden_size=hidden_size
        )
        self.q = Critic(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            hidden_size=hidden_size
        )
        self.q_target = Critic(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            hidden_size=hidden_size
        )
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.q_target.load_state_dict(self.q.state_dict())

    def soft_update_target(self, net, net_target):
        """Update the weights in target network using a weighted sum.
        w_target := tau * w + (1 - tau) * w_target, s.t., tau << 1
        Args:
            net: the policy/q network
            net_target: the target policy/q network
        """
        for param, param_target in zip(net.parameters(), net_target.parameters()):
            param_target.data.copy_(
                self.hparams.tau * param + (1.0 - self.hparams.tau) * param_target.data
            )

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.policy(x)
        return output

    def loss(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the loss for SAC which contains a total of 3 losses.
        Args:
            batch: a batch of states, actions, rewards, dones, and next states
        """
        states, actions, rewards, dones, next_states = batch
        rewards = rewards.unsqueeze(-1)
        dones = dones.float().unsqueeze(-1)

        with torch.no_grad():
            # Select action according to policy
            next_actions = self.policy_target(next_states)

            # Compute the next Q-values: min over all critics targets
            next_q_values = self.q_target(next_states, next_actions).unsqueeze(-1)
            target_q_values = rewards + (1 - dones) * self.hparams.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        estimated_q_values = self.q(states, actions).unsqueeze(-1)

        # Compute critic loss
        critic_loss = F.mse_loss(estimated_q_values, target_q_values)

        # Compute actor loss
        actor_loss = -self.q(states, self.policy(states)).mean()

        return actor_loss, critic_loss

    def training_step(self, batch: Tuple[Tensor, Tensor], _):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.
        Args:
            batch: current mini batch of replay data
            _: batch number, not used
        """
        policy_optim, q_optim = self.optimizers()
        policy_loss, q_loss = self.loss(batch)

        policy_optim.zero_grad()
        self.manual_backward(policy_loss)
        policy_optim.step()

        q_optim.zero_grad()
        self.manual_backward(q_loss)
        q_optim.step()

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.soft_update_target(self.policy, self.policy_target)
            self.soft_update_target(self.q, self.q_target)

        self.log_dict(
            {
                "total_reward": float(self.total_rewards[-1]),
                "avg_reward": float(self.avg_rewards),
                "policy_loss": float(policy_loss),
                "q_loss": float(q_loss),
                "episodes": float(self.done_episodes),
                "episode_steps": float(self.total_episode_steps[-1]),
            }
        )

    def validation_step(self, batch: Tuple[List[float]], _) -> Dict[str, Tensor]:
        """Evaluate the agent for N episodes."""
        total_val_rewards = batch.clone()
        avg_val_reward = torch.mean(total_val_rewards)
        self.log("avg_val_reward", float(avg_val_reward))
        return {"avg_val_reward": avg_val_reward}

    def _train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.hparams.replay_size, self.hparams.n_steps)
        self.populate(self.hparams.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.generate_batch)
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size
        )

    def _val_dataloader(self) -> DataLoader:
        """Initialize the dataset for collecting validation experiences."""
        self.val_dataset = ExperienceSourceDataset(self.run_n_episodes)
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.n_val_episodes
        )

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._train_dataloader()

    def val_dataloader(self) -> DataLoader:
        """Get validation loader."""
        return self._val_dataloader()

    def configure_optimizers(self) -> Tuple[Optimizer]:
        """Initialize Adam optimizer."""
        policy_optim = optim.Adam(self.policy.parameters(), self.hparams.policy_learning_rate)
        q_optim = optim.Adam(self.q.parameters(), self.hparams.q_learning_rate)
        return policy_optim, q_optim

    @staticmethod
    def make_environment(
        seed: Optional[int] = None,
        backbone: Optional[str] = None,
        voc_transforms: Optional[Callable] = None,
        corruption_transforms: Optional[Callable] = None,
        root: Optional[str] = None,
        year: Optional[str] = "2012",
        image_set: Optional[str] = "train",
        beta: Optional[float] = 0.9
    ) -> Env:
        """Initialise SIFT environment.

        Args:
            seed: value to seed the environment RNG for reproducibility
        Returns:
            gym environment
        """
        seed_everything(seed=seed)
        backbone = instantiate(backbone)
        voc_transforms = instantiate(voc_transforms)
        corruption_transforms = instantiate(corruption_transforms)
        image_dataloader = VOCCorruptionDataset(
            root=root,
            year=year,
            image_set=image_set,
            transforms=voc_transforms,
            corruption_transforms=corruption_transforms
        )
        env = SIFTEnv(
            backbone=backbone,
            dataloader=image_dataloader,
            beta=beta
        )

        return env