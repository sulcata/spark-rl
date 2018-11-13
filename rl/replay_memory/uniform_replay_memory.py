import torch

from ..utils import CircularQueue
from . import ReplayMemory


class UniformReplayMemory(ReplayMemory):
    def __init__(self, capacity, replacement=True):
        super().__init__(capacity, replacement, prioritized=False)
        self._memory = CircularQueue(capacity)

    def __getitem__(self, index):
        return self._memory[index]

    def __len__(self):
        return len(self._memory)

    def append(self, item):
        self._memory.append(item)

    def sample(self, k):
        n = len(self)
        indices = torch.randint(n, size=(k,), dtype=torch.long) \
                  if self.replacement else \
                  torch.multinomial(torch.ones(n), k, replacement=True)
        return [self[i] for i in indices]
