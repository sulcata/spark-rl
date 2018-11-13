import gym

import torch
from torch import nn, optim

from rl.replay_memory import UniformReplayMemory
from rl.nn import Dueling, NoisyLinear, ResidualBlock
from rl.agents import DQN
from rl.policy import NoisyNetworkPolicy
from rl.env import GymEnv


class Agent(DQN):
    def __init__(self, nb_actions):
        feature_width = 48
        advantage_width = 48
        value_width = 48
        network = nn.Sequential(
            nn.Linear(4, feature_width),
            nn.ELU(),
            nn.Linear(feature_width, feature_width),
            nn.ELU(),
            Dueling(
                nn.Sequential(
                    NoisyLinear(advantage_width, advantage_width),
                    nn.ELU(),
                    NoisyLinear(advantage_width, nb_actions)
                ),
                nn.Sequential(
                    nn.Linear(value_width, value_width),
                    nn.ELU(),
                    nn.Linear(advantage_width, 1)
                )
            )
        )
        memory = UniformReplayMemory(capacity=5000)
        super().__init__(network=network,
                         batch_size=128,
                         train_policy=NoisyNetworkPolicy(),
                         optimizer=optim.Adam(network.parameters()),
                         memory=memory,
                         target_update_frequency=55,
                         gamma=0.999)

    def preprocess_state(self, state):
        return state.float()


def create_example():
    env = GymEnv(gym.make('CartPole-v0'), max_steps=200)
    agent = Agent(env.nb_actions)
    settings = {}
    return agent, env, settings
