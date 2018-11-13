import torch

from . import Agent
from ..policy import StochasticPolicy


class Stochastic(Agent):
    def __init__(self):
        super().__init__(StochasticPolicy(), StochasticPolicy())
