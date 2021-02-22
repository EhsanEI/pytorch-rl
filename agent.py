import utils
import numpy as np
import torch
import torch.nn.functional as F


class QNet(torch.nn.Module):
    def __init__(self, config):
        super(QNet, self).__init__()

        torch.manual_seed(config['seed'])

        widths = [config['input_dim']]
        for _ in range(config['depth'] - 1):
            widths.append(config['width'])
        widths.append(config['num_actions'])

        self.layers = [torch.nn.Linear(widths[i], widths[i+1])
                       for i in range(len(widths)-1)]

        for layer in self.layers:
            torch.nn.init.orthogonal_(layer.weight)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class ExpSarsaAgent():
    '''Expected Sarsa Network with replay buffer, softmax policy, and Adam.
    All methods assume mini-batches.
    '''

    def __init__(self, config):
        self.config = config

        buffer_config = {
            'seed': config['seed'],
            'size': config['buffer_size']
        }
        self.buffer = utils.ReplayBuffer(config=buffer_config)

        self.rng = np.random.default_rng(config['seed'])

        net_config = {
            'seed': config['seed'],
            'depth': config['network_depth'],
            'width': config['network_width'],
            'num_actions': config['num_actions'],
            'input_dim': config['input_dim']
        }
        self.qnet = QNet(net_config)

    def reset(self):
        pass

    def step(self, x, a, r, gamma, xp):
        pass

    def action(self, x):
        pass
