import numpy as np
import gym
import agent


seed = 0
gamma = 0.99
env = gym.make('LunarLander-v2')
env.seed(seed)

agent_config = {
    'seed': seed,
    'input_dim': env.observation_space.shape[0],
    'num_actions': env.action_space.n,
    'network_depth': 2,
    'network_width': 256,
    'updates_per_step': 4,
    'step_size': 1e-3,
    'buffer_size': 50000,
    'batch_size': 8,
    'softmax_tau': 1e-3
}
esagent = agent.ExpSarsaAgent(agent_config)

returns = []
ret = 0.
x = env.reset()
while len(returns) < 2000:
    a = esagent.act(x[np.newaxis, :])
    xp, r, done, _ = env.step(a)
    ret += r
    if not done:
        esagent.step(x, a, r, gamma, xp)
    else:
        print('episode:', len(returns), '\treturn:', ret)
        returns.append(ret)
        ret = 0.
        xp = env.reset()
        esagent.step(x, a, r, 0.0, xp)
    x = xp
