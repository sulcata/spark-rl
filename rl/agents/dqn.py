from copy import deepcopy

import torch

from . import Agent
from ..nn import huber_loss
from ..policy import EpsilonGreedyPolicy, GreedyPolicy

from ..utils import CircularQueue, Transition


class DQN(Agent):
    def __init__(self, network, optimizer, memory, batch_size,
                 gamma=0.999,
                 n_steps=1,
                 train_policy=EpsilonGreedyPolicy(0.1),
                 eval_policy=GreedyPolicy(),
                 delta_clip=2., warm_up=0,
                 target_update_frequency=0):
        super().__init__(train_policy, eval_policy, gamma)
        self.online_network = network
        self.target_network = network
        self.optimizer = optimizer
        self.memory = memory
        self.batch_size = batch_size
        self.delta_clip = delta_clip
        self.warm_up = warm_up

        self.n_steps = n_steps
        self._n_step_buffer = CircularQueue(n_steps) if n_steps > 1 else None

        self.target_update_frequency = target_update_frequency
        self.double_network = self.target_update_frequency > 0
        if self.double_network:
            self.target_network = deepcopy(network)

    def _sample_memory(self):
        batch, indices, weights = None, None, None
        if self.memory.prioritized:
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
        # Convert to batch tensors
        batch_tensors = self.transitions_to_tensors(batch)
        return batch_tensors, indices, weights

    def q_values(self, state):
        state = self.state_to_tensor([state])
        with torch.no_grad():
            return self.online_network(state).squeeze(0)

    def _q(self, state, action):
        return self.online_network(state).gather(1, action)

    def _q_target(self, next_state, reward, done):
        action = self.online_network(next_state).argmax(1, keepdim=True)
        q_next = (1 - done) * self.target_network(next_state).gather(1, action)
        return reward + (self.gamma ** self.n_steps) * q_next

    def loss(self, state, action, next_state, reward, done):
        q = self._q(state, action)
        with torch.no_grad():
            q_target = self._q_target(next_state, reward, done)
        td_error = q_target - q
        return huber_loss(td_error, delta=self.delta_clip, reduction='none')

    def replay(self):
        batch, indices, weights = self._sample_memory()
        state, action, next_state, reward, done = batch

        loss = self.loss(state, action, next_state, reward, done)

        # Provide
        if self.memory.prioritized:
            priorities = loss.detach().abs().squeeze().cpu()
            self.memory.batch_update(indices, priorities)
            loss *= weights.unsqueeze(1)

        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def before_train(self):
        self.online_network.train()
        self.target_network.train()

    def _add_to_memory(self, new_transition):
        transition = new_transition
        if self.n_steps > 1:
            buffer = self._n_step_buffer
            buffer.append(new_transition)
            buffer_len = len(buffer)
            if not new_transition.done and buffer_len < self.n_steps:
                return
            G = sum(buffer[i].reward * self.gamma ** i for i in range(buffer_len))
            old_transition = self._n_step_buffer[0]
            transition = Transition(old_transition.state,
                                    old_transition.action,
                                    new_transition.next_state,
                                    G,
                                    new_transition.done)
            if new_transition.done:
                self._n_step_buffer.clear()

        if self.memory.prioritized:
            tensors = self.transitions_to_tensors([transition])
            state, action, next_state, reward, done = tensors
            with torch.no_grad():
                q = self.q(state, action)
                q_target = self.q_target(next_state, reward, done)
                priority = self.loss(q, q_target).mean().abs().item()
            self.memory.append(transition, priority)
        else:
            self.memory.append(transition)

    def train_after_step(self, info):
        self._add_to_memory(info.transition)
        if self.memory.can_sample(self.batch_size) and info.total_steps >= self.warm_up:
            self.replay()
            if self.double_network and info.total_steps % self.target_update_frequency == 0:
                state_dict = self.online_network.state_dict()
                self.target_network.load_state_dict(state_dict)

    def before_eval(self):
        self.online_network.eval()
        self.target_network.eval()

    def after_eval(self):
        self.online_network.train()
        self.target_network.train()
