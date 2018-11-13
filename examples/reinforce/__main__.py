import statistics

from . import reinforce


agent, env, settings = reinforce.create_example()
agent.train(env, 1750)
info = agent.eval(env, 100)

print(info.episode_rewards)
print(statistics.mean(info.episode_rewards))
