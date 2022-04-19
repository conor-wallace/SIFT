import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from sift.data import RLDataset


class BaseLightning(pl.LightningModule):
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

        # Train Params
        self.batch_size = cfg.train.batch_size
        self.lr = cfg.train.lr
        self.gamma = cfg.train.gamma
        self.sync_rate = cfg.train.sync_rate
        # Greedy Policy Params
        self.eps_start = cfg.train.policy.eps_start
        self.eps_end = cfg.train.policy.eps_end
        self.eps_last_frame = cfg.train.policy.eps_last_frame
        # Episode Params
        self.episode_length = cfg.data.buffer.episode_length
        self.total_reward = 0
        self.episode_reward = 0

        self.populate(cfg.data.buffer.warm_start_steps)

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for storing experiences"""
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
