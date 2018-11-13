import math

import torch

from ..utils import CircularQueue, coerce_callable
from . import ReplayMemory


class ProportionalReplayMemory(ReplayMemory):
    def __init__(self, capacity, alpha, beta, epsilon=1e-7):
        super().__init__(capacity, replacement=True, prioritized=True)
        self.alpha = alpha
        self.beta = coerce_callable(beta)
        self.epsilon = epsilon
        self.samples_taken = 0
        self._memory = CircularQueue(capacity)
        self._tree_depth = math.ceil(math.log(capacity, 2)) + 1
        self._tree_size = 2 ** self._tree_depth - 1
        self._leaf_start = 2 ** (self._tree_depth - 1) - 1
        self._sum_tree = torch.zeros(self._tree_size, dtype=torch.float)

    def __getitem__(self, index):
        return self._memory[index]

    def __len__(self):
        return len(self._memory)

    def append(self, item, priority):
        self.update(len(self._memory) + self._memory.position, priority)
        self._memory.append(item)

    def sample(self, k):
        sum = self._sum_tree[0]
        indices = torch.zeros(k, dtype=torch.long)
        rand = torch.empty(k, dtype=torch.float).uniform_(0, sum)
        for _ in range(self._tree_depth - 1):
            # Update tree indices to point to the left child
            indices.mul_(2).add_(1)
            left_values = self._sum_tree.take(indices)
            # Determine which path to take based on rand's comparison to left
            go_right = rand > left_values
            # Adding one will change the index to go right if cond is true
            indices.add_(go_right.long())
            # Subtract
            rand.sub_(go_right.float() * left_values)
        # w_i = (1/P(i) * 1/N) ** beta
        n = len(self)
        beta = self.beta(self.samples_taken)
        weights = self._sum_tree.take(indices).mul_(n / sum).pow_(-beta)
        # Convert tree index to list index
        indices.sub_(self._leaf_start)
        sample = [self[i] for i in indices]
        self.samples_taken += 1
        return sample, indices, weights

    def update(self, index, priority):
        node_index = self._leaf_start + index
        self._sum_tree[node_index] = (priority + self.epsilon) ** self.alpha
        while node_index > 0:
            node_index = (node_index - 1) // 2
            left_value = self._sum_tree[2 * node_index + 1]
            right_value = self._sum_tree[2 * node_index + 2]
            self._sum_tree[node_index] = left_value + right_value

    def batch_update(self, indices, priorities):
        node_indices = self._leaf_start + indices
        self._sum_tree[node_indices] = priorities
        for _ in range(self._tree_depth - 1):
            node_indices.sub_(1).div_(2)
            left_values = self._sum_tree.take(2 * node_indices + 1)
            right_values = self._sum_tree.take(2 * node_indices + 2)
            self._sum_tree[node_indices] = left_values + right_values
