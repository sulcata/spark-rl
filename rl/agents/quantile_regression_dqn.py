import torch

from . import DQN
from ..nn import quantile_huber_loss


class QuantileRegressionDQN(DQN):
    def __init__(self, *args, atoms=51, **kwargs):
        super().__init__(*args, **kwargs)
        self.atoms = atoms
        low = 0.5 / atoms
        high = 1 - 0.5 / atoms
        self._quantile_midpoints = torch.linspace(low, high, atoms) \
                                        .view(1, -1, 1)

    def q_values(self, state):
        state = self.state_to_tensor([state])
        with torch.no_grad():
            return self.online_network(state).mean(-1).squeeze(0)

    def _q_quantiles(self, state, action):
        action = action.view(-1, 1, 1).expand(-1, -1, self.atoms)
        return self.online_network(state).gather(1, action).squeeze(1)

    def _q_target_quantiles(self, next_state, reward, done):
        action = self.online_network(next_state) \
                     .mean(-1).argmax(1) \
                     .view(-1, 1, 1) \
                     .expand(-1, -1, self.atoms)
        q_next = (1 - done) * self.target_network(next_state) \
                                  .gather(1, action) \
                                  .view(-1, self.atoms)
        return reward + (self.gamma ** self.n_steps) * q_next

    def loss(self, state, action, next_state, reward, done):
        q = self._q_quantiles(state, action)
        with torch.no_grad():
            q_target = self._q_target_quantiles(next_state, reward, done)
        tau_i = self._quantile_midpoints
        theta_i = q.unsqueeze(-1)
        T_theta_j = q_target.unsqueeze(1)
        td_error = T_theta_j - theta_i
        loss = quantile_huber_loss(error, tau_i, self.delta_clip, reduction='none')
        return loss.mean(1).sum(1, keepdim=True)
