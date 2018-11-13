import gym

import torch
from torch import nn, optim

from rl.agents import REINFORCE
from rl.env import GymEnv
from rl import nn as rlnn


class Agent(REINFORCE):
    def __init__(self, nb_actions):
        width = 48
        feature = nn.Sequential(
            nn.Linear(2, width),
            nn.ELU(),
        )
        policy = nn.Sequential(
            nn.Linear(width, nb_actions),
        )
        value = nn.Sequential(
            nn.Linear(width, 1),
        )
        params = [
            {'params': feature.parameters()},
            {'params': policy.parameters()},
            {'params': value.parameters()},
        ]
        optimizer = optim.Adam(params)
        super().__init__(policy=policy, value=value, feature=feature,
                         optimizer=optimizer, gamma=1.)

    def preprocess_state(self, state):
        return state.float()


def create_example():
    env = GymEnv(gym.make('MountainCar-v0'), max_steps=200)
    agent = Agent(env.nb_actions)
    settings = {}
    return agent, env, settings
