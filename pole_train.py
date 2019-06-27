import gym, torch, torch.optim
import numpy as np, time
from itertools import count
import matplotlib.pyplot as plt

from pole_net import Q_net


class ReplayBuffer:
    def __init__(self, max_size, keys):
        self.dict = {}
        for key in keys:
            self.dict[key] = []  # creating empty list for each key
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
N_episodes = 3000  # for how many episodes to train
env_version = 1  # cartpole version 0 or 1
if env_version == 1:
    T_max = 499  # latest step that environment allows, starting from 0
elif env_version == 0:
    T_max = 199
else:
    assert False, "wrong env_version, should be 0 or 1 (integer)"
method = 'double'  # method for evaluating the targets, only double and single(vanilla) are possible
# method = 'single'
Size_replay_buffer = 100000  # in time steps
eps_start = 1  # eps for epsilon greedy algorithm
eps_end = 0.01
eps_anneal_period = 10000  # simple linear annealing
Size_minibatch = 128
net_update_period = 1000  # after how many minibatches should the target computing net be updated
gamma = 0.99
l2_regularization = 0  # L2 regularization coefficient
net_save_path = 'net_cartpole-v{}_{}DQN.pth'.format(env_version, method)
plot_save_path = 'running_score_cartpole-v{}_{}DQN.png'.format(env_version, method)
device = "cpu"

net = Q_net()  # net that determines policy to follow (except when randomly choosing actions during training)
net.to(device)
net_ = Q_net()  # net that computes the targets
net_.to(device)
net_.load_state_dict(net.state_dict())  # copying all network parameters
net_.eval()
# loss_function = torch.nn.MSELoss()
loss_function = torch.nn.SmoothL1Loss()  # Huber loss
# optimizer = torch.optim.RMSprop(net.parameters(), weight_decay=l2_regularization)
optimizer = torch.optim.Adam(net.parameters(), weight_decay=l2_regularization)

replay_buffer = ReplayBuffer(Size_replay_buffer, keys=['x', 'a', 'r', 'x_next'])

env = gym.make('CartPole-v'+str(env_version))
eps = eps_start
running_score_history = []

backprops_total = 0  # to track when to update the target net

# running_X is computed as: running_X = 0.99*running_X+0.01*X
running_loss, running_score = 0, 0  # score is the sum of rewards achieved in one episode
running_score_best = 0  # network will be saved only if it exceeds previous best running_score


s_cur = env.reset()
s_prev = s_cur
score = 0  # score per episode
t0 = time.time()
for ep in range(N_episodes):
    for step in range(T_max+2):
        # choose an action:
        x = torch.from_numpy(np.concatenate((s_cur, s_cur-s_prev))).float()
        x = x.to(device)  # input to network
        if np.random.rand() < eps:
            action = np.random.randint(2)
        else:
            net.eval()
            q = net(x.view(1, -1))
            action = np.argmax(q.detach().cpu().numpy())

        s_next, r, done, _ = env.step(action)

        if done:
            if step != T_max:
                r = 0  # will be used during training to detect the terminal step
            else:
                # unnecessary, but I found it to give a bit better results compared to r=0 for any terminal step
                r = 1  # if game stopped due to time steps limitation of environment - count like it didn't
        score += r
        # store the experience
        x_next = torch.from_numpy(np.concatenate((s_next, s_next-s_cur))).float()
        replay_buffer.append((x, action, r, x_next))
        if done:
            running_score = score if running_score == 0 else 0.99*running_score + 0.01*score
            running_score_history.append(running_score)
            score = 0
            s_cur = env.reset()
            s_prev = s_cur
        else:
            s_prev = s_cur
            s_cur = s_next

        if eps > eps_end:  # annealing
            eps -= (eps_start-eps_end)/eps_anneal_period

        # train on one minibatch:
        if len(replay_buffer) < Size_minibatch:
            continue
        net.train()
        minibatch_ids = np.random.choice(len(replay_buffer), Size_minibatch)
        minibatch = replay_buffer.get(minibatch_ids)
        xs, actions, rs, next_xs = minibatch.values()
        xs = torch.stack(xs).to(device)  # list of tensors -> tensor
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

        Q_target[final_state_ids] = 0  # terminal states should have V(s) = max(Q(s,a)) = 0
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
            # if running_score > running_score_best:  ## useless
            net_.load_state_dict(net.state_dict())
            # else:  ## useless
            #     net.load_state_dict(net_.state_dict())  ## useless
        ep_played = ep + 1
        if done and ep_played % 100 == 0:
            print("ep: {}, buf_len: {}, eps: {:.3f}, time: {:.2f}s, running_loss: {:.3f}, running_score: {:.1f}".
                  format(ep_played, len(replay_buffer), eps, time.time()-t0,
                                                                           running_loss, running_score))
        if done and ep_played % 100 == 0 and running_score > running_score_best:
            torch.save(net.state_dict(), net_save_path)
            running_score_best = running_score
            print("net saved to '{}'".format(net_save_path))

        if done and ep_played % 100 == 0:
            plt.close('all')
            plt.plot(running_score_history)
            plt.xlabel('episodes')
            plt.ylabel('running score')
            plt.savefig(plot_save_path)
        if done:
            break
env.close()
