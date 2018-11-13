from collections.abc import Sequence


class CircularQueue(Sequence):
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0
        self._length = 0
        self._data = [None] * capacity

    def __getitem__(self, index):
        if index > self._length:
            raise IndexError('CircularQueue index out of range')
        return self._data[(self.position + index) % self.capacity]

    def __len__(self):
        return self._length

    def append(self, item):
        self._data[(self.position + self._length) % self.capacity] = item
        if self._length < self.capacity:
            self._length += 1
        else:
            self.position = (self.position + 1) % self.capacity

    def clear(self):
        self.position = 0
        self._length = 0
        self._data = [None] * self.capacity
