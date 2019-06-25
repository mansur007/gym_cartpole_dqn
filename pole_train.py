import gym, torch, torch.optim
import numpy as np, time
from itertools import count
import matplotlib.pyplot as plt

from pole_net import Q_net


class ReplayBuffer:
    def __init__(self, max_size, keys):
        self.dict = {}
        for key in keys:
            self.dict[key] = []
        self.max_size = max_size

    def append(self, sample):
        for i, key in enumerate(self.dict.keys()):
            self.dict[key].append(sample[i])
        if len(self) > self.max_size:
            for key in self.dict.keys():
                self.dict[key].pop(0)

    def __len__(self):
        return len(self.dict['x'])

    def get(self, ids):
        sub_dict = {}
        for key in self.dict.keys():
            sub_dict[key] = [self.dict[key][i] for i in ids]
        return sub_dict


## hyperparameters
method = 'double'
# method = 'single'
Size_replay_buffer = 100000

eps_start = 1  # eps for epsilon greedy algorithm
eps_end = 0.05
eps_anneal_period = 2000
N_epochs = 1
Size_minibatch = 128
net_update_period = 2000
gamma = 0.999
net_save_path = 'net.pth'
plot_save_path = 'running_score.png'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("device: ", device)

net = Q_net()
net.to(device)
net_ = Q_net()
net_.to(device)
net_.load_state_dict(net.state_dict())
net_.eval()
# loss_function = torch.nn.MSELoss()
loss_function = torch.nn.SmoothL1Loss()
# optimizer = torch.optim.RMSprop(net.parameters())
optimizer = torch.optim.Adam(net.parameters(), weight_decay=0)

replay_buffer = ReplayBuffer(Size_replay_buffer, keys=['x', 'a', 'r', 'x_next'])
env = gym.make('CartPole-v0')
eps = eps_start

running_score_history = []

t0 = time.time()
backprops_total = 0  # to track when to update the stable net

running_loss, running_score = 0, 0
running_score_best = 0
# collect experience
s_cur = env.reset()
s_prev = s_cur
score = 0  # score per episode
ep_played = 0
for step in count():
    ## make an action:
    # env.render()
    x = torch.from_numpy(np.concatenate((s_cur, s_prev))).float()
    x = x.to(device)
    if np.random.rand() < eps:
        action = np.random.randint(2)
    else:
        net.eval()
        q = net(x.view(1, -1))
        action = np.argmax(q.detach().cpu().numpy())
    s_next, r, done, _ = env.step(action)
    if done:
        r = 0
    score += r
    # store the experience
    x_next = torch.from_numpy(np.concatenate((s_next, s_cur))).float()
    replay_buffer.append((x, action, r, x_next))
    if done:
        running_score = score if running_score == 0 else 0.99*running_score + 0.01*score
        score = 0
        s_cur = env.reset()
        s_prev = s_cur
        ep_played += 1
    else:
        s_prev = s_cur
        s_cur = s_next

    if eps > eps_end:
        eps -= (eps_start-eps_end)/eps_anneal_period

    ## train:
    if len(replay_buffer) < Size_minibatch:
        continue
    net.train()
    minibatch_ids = np.random.choice(len(replay_buffer), Size_minibatch)
    minibatch = replay_buffer.get(minibatch_ids)
    xs, actions, rs, next_xs = minibatch.values()
    xs = torch.stack(xs).to(device)
    next_xs = torch.stack(next_xs).to(device)
    rs = np.array(rs)
    final_state_ids = np.nonzero(rs == 0)  # will be needed to calculate targets for terminal states properly
    rs = torch.from_numpy(rs).float()

    if method == 'double':
        # finding targets by double DQN method
        with torch.no_grad():
            net.eval()
            Q_next = net(next_xs)
            Q_next_ = net_(next_xs)
        net.train()
        optimizer.zero_grad()
        Q = net(xs)
        Q_next_max, Q_next_argmax = torch.max(Q_next, 1)
        Q_target = torch.gather(Q_next_, 1, Q_next_argmax.view(-1, 1)).squeeze()
    else:
        # finding targets by vanilla method
        with torch.no_grad():
            Q_next_ = net_(next_xs)
        optimizer.zero_grad()
        Q = net(xs)
        Q_next_max, Q_next_argmax = torch.max(Q_next_, 1)
        Q_target = Q_next_max

    Q_target[final_state_ids] = 0
    targets = (rs.to(device) + gamma*Q_target).to(device)
    # backprop only on actions that actually occured at corresponding states
    actions = torch.tensor(actions).view(-1, 1)
    Q_relevant = torch.gather(Q, 1, actions.to(device)).squeeze()
    loss = loss_function(Q_relevant, targets)
    loss.backward()
    optimizer.step()
    running_loss = loss.item() if running_loss == 0 else 0.99*running_loss + 0.01*loss.item()

    backprops_total += 1
    if backprops_total % net_update_period == 0:
        net_.load_state_dict(net.state_dict())

    running_score_history.append(running_score)
    if done and ep_played%100==0:
        print("ep: {}, eps: {:.3f}, time: {:.2f}s, running_loss: {}, running_score: {}".format(ep_played, eps, time.time()-t0,
                                                                       running_loss, running_score))
    if done and ep_played % 100 == 0 and running_score > running_score_best:
        torch.save(net.state_dict(), net_save_path)
        running_score_best = running_score
        print("net saved")

    if done and ep_played % 100 == 0:
        plt.close('all')
        plt.plot(running_score_history)
        plt.savefig(plot_save_path)

env.close()