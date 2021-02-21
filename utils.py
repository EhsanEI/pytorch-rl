from collections import deque


class ReplayBuffer():

    def __init__(self, seed):
        self.queue = deque()

    def push(self, x, a, r, gamma, xp):
        pass

    def pop(self):
        pass

    def sample(self, size):
        pass
