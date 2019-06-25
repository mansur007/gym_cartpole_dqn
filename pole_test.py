from pole_net import Q_net
import torch, gym
import numpy as np


net_load_path = 'net.pth'
net = Q_net()
net.load_state_dict(torch.load(net_load_path))
net.eval()

env = gym.make('CartPole-v0')

for ep in range(10):
    s_cur = env.reset()
    s_prev = s_cur
    r_sum = 0
    while True:
        env.render()
        x = torch.from_numpy(np.concatenate((s_cur, s_prev))).float()
        q = net(x.view(1,-1)).squeeze()
        qmax, a = torch.max(q, 0)
        a = a.item()
        s_prev = s_cur
        s_cur, r, done, _ = env.step(a)
        r_sum += r
        # time.sleep(0.1)
        if done:
            break
    print("ep {}, r_sum: {}".format(ep, r_sum))
env.close()
