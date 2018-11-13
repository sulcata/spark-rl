from abc import abstractmethod
from copy import deepcopy

import torch

from ..utils import Observable


def _noop(*args, **kwargs):
    pass


class Agent():
    def __init__(self, train_policy, eval_policy, gamma=1.):
        self.gamma = gamma

        self.train_policy = train_policy
        self.train_events = Observable([
            'before_episode',
            'after_episode',
            'before_step',
            'after_step',
        ])
        self.train_events.subscribe('before_episode', train_policy.before_episode)
        self.train_events.subscribe('after_episode', train_policy.after_episode)
        self.train_events.subscribe('before_step', train_policy.before_step)
        self.train_events.subscribe('after_step', train_policy.after_step)

        self.eval_policy = eval_policy
        self.eval_events = Observable([
            'before_episode',
            'after_episode',
            'before_step',
            'after_step',
        ])
        self.eval_events.subscribe('before_episode', eval_policy.before_episode)
        self.eval_events.subscribe('after_episode', eval_policy.after_episode)
        self.eval_events.subscribe('before_step', eval_policy.before_step)
        self.eval_events.subscribe('after_step', eval_policy.after_step)

    def preprocess_state(self, state):
        return state

    def preprocess_reward(self, reward):
        return reward

    def state_to_tensor(self, state):
        return self.preprocess_state(torch.as_tensor(state))

    def reward_to_tensor(self, reward):
        d = torch.get_default_dtype()
        reward_tensor = torch.as_tensor(reward, dtype=d).unsqueeze(1)
        return self.preprocess_reward(reward_tensor)

    def transitions_to_tensors(self, transitions):
        d = torch.get_default_dtype()
        state, action, next_state, reward, done = zip(*transitions)
        state = self.state_to_tensor(state)
        action = torch.as_tensor(action, dtype=torch.long).unsqueeze(1)
        next_state = self.state_to_tensor(next_state)
        reward = self.reward_to_tensor(reward)
        done = torch.as_tensor(done, dtype=d).unsqueeze(1)
        return state, action, next_state, reward, done

    def train_after_step(self, info):
        pass

    def train_after_episode(self, info):
        pass

    def before_train(self):
        pass

    def after_train(self):
        pass

    def train(self, env, nb_episodes,
              before_action=_noop, after_action=_noop,
              before_episode=_noop, after_episode=_noop):
        events = deepcopy(self.train_events)
        events.subscribe('before_step', before_action)
        events.subscribe('after_step', after_action)
        events.subscribe('before_episode', before_episode)
        events.subscribe('after_episode', after_episode)
        info = SessionInfo()
        self.before_train()
        for episode in range(nb_episodes):
            info._begin_episode()
            events.dispatch('before_episode', self, info)
            env.reset()
            while True:
                events.dispatch('before_step', self, info)
                action = self.train_policy.select_action(self, env, info)
                info._select_action(action)
                transition = env.step(action)
                info._take_step(transition)
                events.dispatch('after_step', self, info)
                self.train_after_step(info)
                if transition.done:
                    break
            info._end_episode()
            events.dispatch('after_episode', self, info)
            self.train_after_episode(info)
        self.after_train()
        return info

    def before_eval(self):
        pass

    def after_eval(self):
        pass

    def eval(self, env, nb_episodes,
             before_action=_noop, after_action=_noop,
             before_episode=_noop, after_episode=_noop):
        events = deepcopy(self.eval_events)
        events.subscribe('before_step', before_action)
        events.subscribe('after_step', after_action)
        events.subscribe('before_episode', before_episode)
        events.subscribe('after_episode', after_episode)
        info = SessionInfo()
        self.before_eval()
        for episode in range(nb_episodes):
            info._begin_episode()
            events.dispatch('before_episode', self, info)
            env.reset()
            while True:
                events.dispatch('before_step', self, info)
                action = self.eval_policy.select_action(self, env, info)
                info._select_action(action)
                transition = env.step(action)
                info._take_step(transition)
                events.dispatch('after_step', self, info)
                if transition.done:
                    break
            info._end_episode()
            events.dispatch('after_episode', self, info)
        self.after_eval()
        return info


class SessionInfo():
    def __init__(self):
        self._episode = 0
        self._reward = 0.
        self._action = 0
        self._transition = None
        self._step = 0
        self._total_steps = 0
        self._returns = []

    @property
    def episode(self):
        return self._episode

    @property
    def reward(self):
        return self._reward

    @property
    def action(self):
        return self._action

    @property
    def transition(self):
        return self._transition

    @property
    def step(self):
        return self._step

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def returns(self):
        return self._returns.copy()

    def _begin_episode(self):
        self._reward = 0.
        self._step = 0

    def _end_episode(self):
        self._episode += 1
        self._returns.append(self.reward)

    def _select_action(self, action):
        self._action = action

    def _take_step(self, transition):
        self._transition = transition
        self._reward += transition.reward
        self._step += 1
        self._total_steps += 1
