import numpy as np


class ReplayBuffer():
    '''Experience Replay buffer. Implemented as a cyclic array
    of fixed size for efficiency.
    '''

    def __init__(self, config):
        self.max_size = config['size']

        self.array = []
        self.position = 0

        self.rng = np.random.default_rng(config['seed'])

    def push(self, x, a, r, gamma, xp):
        if len(self.array) < self.max_size:
            self.array.append(None)

        self.array[self.position] = (x, a, r, gamma, xp)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        indices = self.rng.integers(len(self.array), size=batch_size)
        return [self.array[i] for i in indices]
