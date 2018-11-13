import gym

import torch
from torch import nn, optim

from rl.policy import EpsilonGreedyPolicy, NoisyNetworkPolicy
from rl.schedule import LinearSchedule
from rl.replay_memory import UniformReplayMemory
from rl import agents, nn as rlnn
from rl.env import GymEnv


class Agent(agents.CategoricalDQN):
    def __init__(self, nb_actions):
        feature_width = 96
        advantage_width = 96
        value_width = 96
        network = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.ELU(),
            nn.Linear(feature_width, feature_width),
            nn.ELU(),
            rlnn.Dueling(
                nn.Sequential(
                    rlnn.NoisyLinear(advantage_width, advantage_width),
                    nn.ELU(),
                    rlnn.NoisyDistributional(advantage_width, nb_actions),
                ),
                nn.Sequential(
                    nn.Linear(value_width, value_width),
                    nn.ELU(),
                    rlnn.Distributional(value_width, 1),
                ),
                distributional=True
            )
        )
        super().__init__(network=network,
                         batch_size=128,
                         train_policy=NoisyNetworkPolicy(),
                         optimizer=optim.Adam(network.parameters()),
                         memory=UniformReplayMemory(capacity=1000),
                         target_update_frequency=50,
                         warm_up=300,
                         gamma=1.)

    def preprocess_state(self, state):
        state = state.float()
        player = (state[:,0] - 12.5) / 8.5
        dealer = (state[:,1] - 5.5) / 4.5
        ace = 2 * state[:,2] - 1
        return torch.stack([player, dealer, ace], dim=1)


def create_example():
    env = GymEnv(gym.make('Blackjack-v0'))
    agent = Agent(env.nb_actions)
    settings = {}
    return agent, env, settings
