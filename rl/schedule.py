from abc import abstractmethod
from collections.abc import Callable
import math


class Schedule(Callable):
    pass


class LinearSchedule(Schedule):
    def __init__(self, value_begin, value_end, step_begin, step_end):
        self.value_begin = value_begin
        self.value_end = value_end
        self.step_begin = step_begin
        self.step_end = step_end
        self._slope = (value_end - value_begin) / (step_end - step_begin)

    def __call__(self, t):
        if t <= self.step_begin:
            return self.value_begin
        if t >= self.step_end:
            return self.value_end
        return self._slope * (t - self.step_begin) + self.value_begin


class ExponentialDecaySchedule(Schedule):
    def __init__(self, value_init, value_min, rate):
        self.value_init = value_init
        self.value_min = value_min
        self.rate = rate

    def __call__(self, t):
        a = self.value_init - self.value_min
        return a * (self.rate ** t) + self.value_min


class LinearCyclicSchedule(Schedule):
    def __init__(self, value1, value2, period, offset=0.):
        self.value1 = value1
        self.value2 = value2
        self.period = period
        self.offset = offset

    def __call__(self, t):
        t = ((t + self.offset) % self.period) * 2 / self.period
        v1 = self.value1
        v2 = self.value2
        if t >= 1:
            v1 = self.value2
            v2 = self.value1
            t -= 1
        return (1 - t) * v1 + t * v2
