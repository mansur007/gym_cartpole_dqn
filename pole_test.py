from pole_net import Q_net
import torch, gym
import time


net_load_path = 'net.pth'
net = Q_net()
net.load_state_dict(torch.load(net_load_path))

env = gym.make('CartPole-v0')

for ep in range(10):
    s = env.reset()
    r_sum = 0
    while True:
        env.render()
        x = torch.from_numpy(s).float()
        q = net(x)
        qmax, a = torch.max(q, 0)
        a = a.item()
        s, r, done, _ = env.step(a)
        r_sum += r
        # time.sleep(0.1)
        if done:
            break
    print("ep {}, r_sum: {}".format(ep, r_sum))
env.close()
