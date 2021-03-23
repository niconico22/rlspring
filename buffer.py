import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buff_size, state_shape , action_shape ,device ):
        self.buff_size = buff_size
        self._p = 0
        self._n = 0
        self.states = torch.empty((buff_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buff_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buff_size, 1), dtype=torch.float, device=device)
        self.terminals = torch.empty((buff_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buff_size, *state_shape), dtype=torch.float, device=device)
        
    def add(self, state, action, next_state, reward, done):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.terminals[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self._p = (self._p + 1) % self.buff_size
        self._n = min(self._n + 1, self.buff_size)

    def sample_buffer(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.next_states[idxes],
            self.rewards[idxes],
            self.terminals[idxes]
        )


