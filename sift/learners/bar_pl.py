from typing import Callable, Optional, Tuple, OrderedDict, List
import logging

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from pl_bolts.losses.rl import dqn_loss
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm
from gym import Env

from pl_bolts.datamodules.experience_source import Experience, ExperienceSourceDataset
from pl_bolts.models.rl.common.agents import ValueAgent
from pl_bolts.models.rl.common.networks import MLP

from sift.learners import BaseLightning
from sift.networks import DQN
from sift.agents import DQNAgent
from sift.data import ReplayBuffer, VOCCorruptionDataset
from sift.envs import BAREnv


class BARLightning(BaseLightning):
    def __init__(
        self,
        seed: Optional[int] = 123,
        backbone: Optional[str] = "resnet50",
        voc_transforms: Optional[Callable] = nn.Identity,
        corruption_transforms: Optional[Callable] = nn.Identity,
        root: Optional[str] = None,
        year: Optional[str] = "2012",
        image_set: Optional[str] = "train",
        download: Optional[bool] = False,
        coefficient: Optional[float] = 0.01,
        beta: Optional[float] = 0.9,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 150000,
        sync_rate: int = 1000,
        gamma: float = 0.99,
        hidden_size: int = 128,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        replay_size: int = 100000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -21,
        batches_per_epoch: int = 1000,
        n_steps: int = 1,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_steps: how many samples do we use to fill our buffer
            at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()

        # Environment
        self.exp = None
        self.env = self.make_environment(
            seed=seed,
            backbone=backbone,
            voc_transforms=voc_transforms,
            corruption_transforms=corruption_transforms,
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            coefficient=coefficient,
            beta=beta
        )

        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n

        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.net = None
        self.target_net = None
        self.build_networks(
            obs_shape=obs_shape,
            n_actions=n_actions,
            hidden_size=hidden_size
        )

        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_frames=eps_last_frame,
        )

        # Hyperparameters
        self.sync_rate = sync_rate
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.batches_per_epoch = batches_per_epoch
        self.n_steps = n_steps

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
            self.agent.play_step(self.net, epsilon=1.0)

    def build_networks(
        self,
        obs_shape: Tuple[int],
        n_actions: int,
        hidden_size: Optional[int] = 128
    ) -> None:
        """Initializes the DQN train and target networks."""
        self.net = MLP(
            input_shape=obs_shape,
            n_actions=n_actions,
            hidden_size=hidden_size
        )
        self.target_net = MLP(
            input_shape=obs_shape,
            n_actions=n_actions,
            hidden_size=hidden_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state x through the network and gets the q_values
        of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def train_batch(
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

            next_state, r, is_done, _ = self.env.step(action[0])

            episode_reward += r
            episode_steps += 1

            exp = Experience(state=self.state, action=action[0], reward=r, done=is_done, new_state=next_state)

            self.agent.update_epsilon(self.global_step)
            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(self.batch_size)

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> OrderedDict:
        """Carries out a single step through the environment to update the
        replay buffer. Then calculates loss based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(
            self.eps_end,
            self.eps_start - self.global_step + 1 / self.eps_last_frame,
        )

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = dqn_loss(batch, self.net, self.target_net, self.gamma)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log("q_loss", loss)
        self.log("total_reward", self.total_reward)
        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.replay_size, self.n_steps)
        self.populate(self.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def make_environment(
        seed: Optional[int] = None,
        backbone: Optional[str] = "resnet50",
        voc_transforms: Optional[Callable] = nn.Identity,
        corruption_transforms: Optional[Callable] = nn.Identity,
        root: Optional[str] = None,
        year: Optional[str] = "2012",
        image_set: Optional[str] = "train",
        download: Optional[bool] = False,
        coefficient: Optional[float] = 0.01,
        beta: Optional[float] = 0.9
    ) -> Env:
        """Initialise BAR environment.

        Args:
            seed: value to seed the environment RNG for reproducibility
        Returns:
            gym environment
        """
        backbone = instantiate(backbone)
        dataloader = VOCCorruptionDataset(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transforms=voc_transforms,
            corruption_transforms=corruption_transforms
        )
        env = BAREnv(
            backbone=backbone,
            dataloader=dataloader,
            coefficient=coefficient,
            beta=beta
        )

        return env