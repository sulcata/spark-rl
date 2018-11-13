import statistics

from . import cartpole_v0


agent, env, settings = cartpole_v0.create_example()
agent.train(env, 400)
info = agent.eval(env, 100)

print(info.returns)
print(statistics.mean(info.returns))
