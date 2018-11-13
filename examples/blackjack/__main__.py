import statistics

import torch

from rl.agents import Agent
from rl.policy import Policy

from . import iqn_bj as bj


torch.manual_seed(314 * 271 * 508)


class BlackjackPolicy(Policy):
    def select_action(self, agent, env, info):
        return int(env.state[0] - env.state[1] <= 5 and env.state[0] <= 16)


class BaselineBlackjackAgent(Agent):
    def __init__(self):
        super().__init__(BlackjackPolicy(), BlackjackPolicy())


agent, env, settings = bj.create_example()

baseline_agent = BaselineBlackjackAgent()
env.seed(314)
baseline_info = agent.eval(env, 50000)
baseline_mean = statistics.mean(baseline_info.returns)
baseline_stdev = statistics.stdev(baseline_info.returns)
print(baseline_mean, baseline_stdev)

env.seed(271)
agent.train(env, 400 + 1280)
env.seed(508)
info = agent.eval(env, 1000)
mean = statistics.mean(info.returns)

print(mean)
print(f'z-score: {(mean - baseline_mean) / baseline_stdev}')
