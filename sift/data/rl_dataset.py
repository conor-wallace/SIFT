from typing import Iterator, Tuple
from torch.utils.data.dataset import IterableDataset

from sift.data.buffer import ReplayBuffer


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated
    with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, batch_size: int = 64) -> None:
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple]:
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, dones, new_states = batch
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
