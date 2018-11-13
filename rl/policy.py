from abc import abstractmethod
import math

import torch

from .utils import coerce_callable


class Policy():
    @abstractmethod
    def select_action(self, agent, env, info):
        pass

    def before_step(self, agent, info):
        pass

    def after_step(self, agent, info):
        pass

    def before_episode(self, agent, info):
        pass

    def after_episode(self, agent, info):
        pass


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        self.epsilon = coerce_callable(epsilon)
        self._epsilon = 0.

    def select_action(self, dqn_agent, env, info):
        if self._epsilon > 0 and torch.rand(1) <= self._epsilon:
            return env.sample_action()
        return dqn_agent.q_values(env.state).argmax().item()

    def before_episode(self, dqn_agent, info):
        self._epsilon = self.epsilon(info.episode)


class GreedyPolicy(Policy):
    def select_action(self, dqn_agent, env, info):
        return dqn_agent.q_values(env.state).argmax().item()


class StochasticPolicy(Policy):
    def select_action(self, agent, env, info):
        return env.sample_action()


class NoisyNetworkPolicy(Policy):
    def select_action(self, dqn_agent, env, info):
        return dqn_agent.q_values(env.state).argmax().item()

    def before_step(self, dqn_agent, info):
        dqn_agent.online_network.apply(_sample_noise)

    def after_step(self, dqn_agent, info):
        dqn_agent.online_network.apply(_sample_noise)
        if dqn_agent.double_network:
            dqn_agent.target_network.apply(_sample_noise)


def _sample_noise(module):
    if hasattr(module, 'sample_noise') and callable(module.sample_noise):
        module.sample_noise()


class BoltzmannPolicy(Policy):
    def select_action(self, agent, env, info):
        probs = agent.probabilities(env.state)
        return torch.multinomial(probs, 1).item()
