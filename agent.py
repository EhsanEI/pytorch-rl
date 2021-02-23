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

        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(widths[i], widths[i+1])
             for i in range(len(widths)-1)])

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
        self.optimizer = torch.optim.Adam(self.qnet.parameters(),
                                          lr=config['step-size'])

    def step(self, x, a, r, gamma, xp):
        self.buffer.push(x, a, r, gamma, xp)
        batch = self.buffer.sample(self.config['batch_size'])
        self._train(batch)

    def _train(self, batch):
        x = np.array([item[0] for item in batch])
        x_tensor = torch.from_array(x).float()
        xp = np.array([item[4] for item in batch])
        xp_tensor = torch.from_array(xp).float()
        qx = self.qnet(x_tensor)

        # TODO: handle termination?
        # TODO: separate target net
        qxp = self.qnet(xp_tensor).detach()

        a = np.array([item[1] for item in batch])
        r = np.array([item[2] for item in batch])
        gamma = np.array([item[3] for item in batch])

        target = qx.detach().clone()

        pip = F.softmax(qxp)  # policy for xp
        bootstrap = (qxp*pip).sum(dim=-1)

        for i in range(target.shape[0]):
            target[i][a[i]] = r[i] + gamma[i] * bootstrap[i]

        loss = torch.nn.MSELoss(qx, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, x):
        value = self.qnet(x)
        return self._softmax_policy(value)

    def _softmax_policy(self, value):
        action = F.softmax(value).argmax(dim=-1)
        return action.detach().cpu().numpy()
