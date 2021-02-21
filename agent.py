import utils
import numpy as np


class ExpSarsaAgent():
    '''Expected Sarsa Network with replay buffer and softmax policy and Adam.
    Methods work with mini-batches.
    '''

    def __init__(self, config):
        self.config = config

        buffer_config = {
            'seed': config['seed'],
            'size': config['buffer_size']
        }
        self.buffer = utils.ReplayBuffer(config=buffer_config)

        self.rng = np.random.default_rng(config['seed'])

        pass

    def reset(self):
        pass

    def step(self, x, a, r, gamma, xp):
        pass

    def action(self, x):
        pass
