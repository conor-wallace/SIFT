from typing import Tuple, OrderedDict, List
import logging

import torch
from torch.optim import Adam, Optimizer
from pl_bolts.losses.rl import dqn_loss
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm

from sift.learners import BaseLightning
from sift.models import DQN
from sift.agents import DQNAgent
from sift.data import ReplayBuffer, VOCCorruptionDataset
from sift.envs import BAREnv


class BARLightning(BaseLightning):
    def __init__(self, cfg: DictConfig) -> None:
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
        self.env = BAREnv(
            backbone=backbone,
            dataloader=dataloader,
            episode_length=cfg.data.buffer.episode_length,
            c=cfg.model.bar.c,
            beta=cfg.model.bar.beta
        )
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(cfg.data.buffer.replay_size)
        self.agent = DQNAgent(self.env, self.buffer)
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
            self.agent.play_step(self.net, epsilon=1.0)

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

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return dqn_loss(batch, self.net, self.target_net, self.gamma)

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
        loss = self.loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        if (self.global_step + 1) % 100 == 0:
            logging.debug("REWARD")
            logging.debug(self.total_reward)

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]
