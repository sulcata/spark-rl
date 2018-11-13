import torch
from torch import autograd as grad
from torch.nn import functional as F

from . import Agent
from ..policy import BoltzmannPolicy
from ..utils import coerce_callable


class REINFORCE(Agent):
    def __init__(self, policy, optimizer,
                 value=0, feature=(lambda x: x),
                 gamma=0.999):
        super().__init__(BoltzmannPolicy(), BoltzmannPolicy(), gamma)
        self.policy = policy
        self.optimizer = optimizer
        self.value = coerce_callable(value)
        self.feature = feature
        self._rollout = []

        def store_transition(_, info):
            self._rollout.append(info.transition)
        self.train_events.subscribe('after_step', store_transition)

        def reset_rollout(_, __):
            self._rollout = []
        self.train_events.subscribe('before_episode', reset_rollout)

    def logits(self, state):
        state = self.state_to_tensor([state]).squeeze(0)
        with torch.no_grad():
            feature = self.feature(state)
            return self.policy(feature)

    def probabilities(self, state):
        logits = self.logits(state)
        return F.softmax(logits, 0)

    def before_train(self):
        self.policy.train()

    def train_after_episode(self, info):
        rollout = self._rollout
        T = len(rollout)
        # Convert to tensors
        state, action, _, reward, _ = self.transitions_to_tensors(rollout)
        reward = reward.squeeze()
        discount = self.gamma ** torch.arange(T, dtype=torch.get_default_dtype())
        for t in range(T):
            G = torch.sum(reward[t:T] * discount[:T-t])
            value = self.value(self.feature(state[t]))
            delta = G - value
            value_loss = -delta * value
            logits = self.policy(self.feature(state[t]))
            log_prob = F.log_softmax(logits, 0).gather(0, action[t])
            policy_loss = -(self.gamma ** t) * delta.detach() * log_prob
            self.optimizer.zero_grad()
            if value_loss.requires_grad:
                grad.backward([value_loss, policy_loss])
            else:
                policy_loss.backward()
            self.optimizer.step()

    def before_eval(self):
        self.policy.eval()

    def after_eval(self):
        self.policy.train()
