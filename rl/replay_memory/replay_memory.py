from abc import abstractmethod
from collections.abc import Sequence


class ReplayMemory(Sequence):
    def __init__(self, capacity, replacement, prioritized):
        self.capacity = capacity
        self.replacement = replacement
        self.prioritized = prioritized

    @abstractmethod
    def append(self, item):
        pass

    @abstractmethod
    def sample(self, k):
        pass

    def update(self, index, priority):
        raise NotImplementedError('update should be overridden')

    def batch_update(self, indices, priorities):
        raise NotImplementedError('batch_update should be overriden')

    def can_sample(self, k):
        return self.replacement or len(self) >= k
