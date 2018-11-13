import torch
from torch.nn import functional as F

from . import DQN


class CategoricalDQN(DQN):
    def __init__(self, *args, atoms=51, v_min=-10.0, v_max=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self._z = torch.linspace(v_min, v_max, atoms).view(1, 1, -1)
        self._dz = (v_max - v_min) / (atoms - 1)

    def q_values(self, state):
        state = self.state_to_tensor([state])
        with torch.no_grad():
            probs = F.softmax(self.online_network(state), -1)
            return self._z.view(1, -1).mul(probs).sum(-1)

    def _q_logits(self, state, action):
        action = action.view(-1, 1, 1).expand(-1, -1, self.atoms)
        return self.online_network(state).gather(1, action).squeeze(1)

    def _q_target_probs(self, next_state, reward, done):
        probs = F.softmax(self.online_network(next_state), -1)
        action = probs.mul(self._z) \
                      .sum(-1).argmax(1) \
                      .view(-1, 1, 1) \
                      .expand(-1, -1, self.atoms)
        q_next_logits = self.target_network(next_state) \
                            .gather(1, action) \
                            .view(-1, self.atoms) \
                            .mul(1 - done)
        q_next_probs = F.softmax(q_next_logits, -1)
        z = (1 - done) * self._z.view(1, -1)
        Tz = reward + (self.gamma ** self.n_steps) * z
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self._dz
        l = b.floor().long()
        u = b.ceil().long()
        l[(u > 0) * (l == u)] -= 1
        u[(l < self.atoms - 1) * (l == u)] += 1
        offset = torch.arange(0, self.batch_size * self.atoms, self.atoms) \
                      .unsqueeze(1).expand(-1, self.atoms)
        m = torch.zeros(self.batch_size, self.atoms)
        m.view(-1) \
         .index_add_(0, (offset + l).view(-1),
                     (q_next_probs * (u.float() - b)).view(-1)) \
         .index_add_(0, (offset + u).view(-1),
                     (q_next_probs * (b - l.float())).view(-1))
        return m

    def loss(self, state, action, next_state, reward, done):
        q_log_probs = F.log_softmax(self._q_logits(state, action), -1)
        q_target_probs = self._q_target_probs(next_state, reward, done)
        return -(q_target_probs * q_log_probs).sum(-1)
