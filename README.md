# Spark-RL
Still very much a WIP. I'll get back to this Eventually™, whenever I have free time.

Maybe I'll even figure out how to work multi-agent algorithms into it and make an env for
playing RBY or GSC.

## Running Examples
* Make sure you've installed PyTorch and Gym
* `python3 -m examples.cartpole`

## Implementations

### Deep-Q Learning
* DQN
    * Double
    * Dueling
    * N-Step
* Categorical DQN (C51)
* Quantile Regression DQN (QR-DQN)
* Implicit Quantile Networks (IQN)

### Replay Memory
* Uniform Replay Memory (with or without replacement)
* Proportional Prioritized Replay Memory (only with replacement)
* REINFORCE (i probably did this slightly wrong, and yet it learns)

### Exploration Policies
* Greedy
* Epsilon-Greedy
* Stochastic
* Noisy Network
* Boltzmann

## Future Plans
* Multi-agent RL (primary, long-term goal)
* Cite specifications and ideas from arXiv papers and other sources
    * Like the asinh-space suggestion from that reddit post on NALU
* Rank-based Prioritized Replay Memory
* Deep Recursive Q Network (DRQN)
* Modern PG and Actor Critic methods
* Replace my Observable implementation with RxPy? Who doesn't want more dependencies?
    * This might be a good idea though.
* Rewrite the whole thing in Rust to add ***fearless concurrency***
    * Jokes aside, it could be worth it for replay memory.
    * Not really a concern until I have all the features I want.
    * This isn't an excuse to learn Rust I swear.

Contains some PyTorch nn modules that aren't directly related to RL, but
are potentially useful or interesting. I might extract them into another repo
later.
