from abc import abstractmethod

from .utils import Transition


_identity = lambda x: x


class Env():
    def __init__(self, state_processor=_identity, reward_processor=_identity):
        self._state = None
        self.state_processor = state_processor
        self.reward_processor = reward_processor

    @property
    def state(self):
        return self._state

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def sample_action(self):
        pass

    @abstractmethod
    def seed(self, seed=None):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @property
    @abstractmethod
    def nb_actions(self):
        pass


class GymEnv(Env):
    def __init__(self, gym_env, max_steps=float('inf'), **kwargs):
        super().__init__(**kwargs)
        self.gym_env = gym_env
        self.max_steps = max_steps
        self._step = -1

    def reset(self):
        self._state = self.state_processor(self.gym_env.reset())
        self._step = 0

    def sample_action(self):
        return self.gym_env.action_space.sample()

    def seed(self, seed=None):
        self.gym_env.seed(seed)

    def step(self, action):
        assert self._step >= 0, \
            "must call reset before stepping in an environment"
        state = self._state
        next_state, reward, done, _ = self.gym_env.step(action)
        next_state = self.state_processor(next_state)
        reward = self.reward_processor(reward)
        self._step += 1
        if self._step >= self.max_steps:
            done = True
            self._state = None
            self._step = -1
        else:
            self._state = next_state
        return Transition(state, action, next_state, reward, done)

    @property
    def nb_actions(self):
        return self.gym_env.action_space.n
