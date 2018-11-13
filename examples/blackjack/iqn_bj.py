import gym

import torch
from torch import nn, optim

from rl.policy import NoisyNetworkPolicy
from rl.replay_memory import UniformReplayMemory
from rl import agents, nn as rlnn
from rl.env import GymEnv


class Agent(agents.IQN):
    def __init__(self, nb_actions):
        feature_width = 96
        advantage_width = 96
        value_width = 96
        sample_embedding_width = 64
        network = rlnn.Sequential(
            rlnn.IQNFeature(
                nn.Sequential(
                    nn.Linear(3, feature_width),
                    nn.ELU(),
                    nn.Linear(feature_width, feature_width),
                    nn.ELU(),
                ),
                nn.Sequential(
                    rlnn.CosineEmbedding(sample_embedding_width),
                    nn.Linear(sample_embedding_width, feature_width),
                    nn.ELU(),
                )
            ),
            nn.Sequential(
                rlnn.NoisyLinear(feature_width, advantage_width),
                nn.ELU(),
                rlnn.NoisyLinear(advantage_width, nb_actions),
            )
        )
        super().__init__(network=network,
                         batch_size=128,
                         train_policy=NoisyNetworkPolicy(),
                         optimizer=optim.Adam(network.parameters()),
                         memory=UniformReplayMemory(capacity=1280),
                         sample_sizes=(32, 32, 32),
                         target_update_frequency=25,
                         warm_up=1280,
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
