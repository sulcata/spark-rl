import torch

from . import DQN
from ..nn import quantile_huber_loss


class IQN(DQN):
    def __init__(self, *args, sample_sizes=(8, 8, 32), **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_sizes = sample_sizes

    def q_values(self, state):
        sample_size = self.sample_sizes[2]
        state = self.state_to_tensor([state])
        tau = torch.empty(sample_size).uniform_(0, 1)
        with torch.no_grad():
            return self.online_network(state, tau).mean(0)

    def _q_samples(self, state, action):
        sample_size = self.sample_sizes[0]
        state = state.unsqueeze(1)
        tau = torch.empty(1, sample_size).uniform_(0, 1)
        action = action.unsqueeze(1).expand(-1, sample_size, -1)
        return tau, self.online_network(state, tau).gather(2, action).squeeze(-1)

    def _q_target_samples(self, next_state, reward, done):
        sample_size = self.sample_sizes[1]
        next_state = next_state.unsqueeze(1)
        action = self._max_q_action(next_state) \
                     .unsqueeze(1).expand(-1, sample_size, -1)
        tau = torch.empty(1, sample_size).uniform_(0, 1)
        q_next = (1 - done) * self.target_network(next_state, tau) \
                                  .gather(2, action).squeeze(-1)
        return reward + (self.gamma ** self.n_steps) * q_next

    def _max_q_action(self, state):
        sample_size = self.sample_sizes[2]
        tau = torch.empty(1, sample_size).uniform_(0, 1)
        return self.online_network(state, tau).mean(1).argmax(1, keepdim=True)

    def loss(self, state, action, next_state, reward, done):
        tau, q_samples = self._q_samples(state, action)
        with torch.no_grad():
            q_target_samples = self._q_target_samples(next_state, reward, done)
        td_error = q_target_samples.unsqueeze(1) - q_samples.unsqueeze(-1)
        tau = tau.unsqueeze(-1)
        loss = quantile_huber_loss(td_error, tau, self.delta_clip, reduction='none')
        return loss.mean(-1).sum(1)
