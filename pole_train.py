import gym, torch, torch.optim
import numpy as np, time
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
Size_replay_buffer = 1000000
N_iters = 1000
N_steps_per_iter = 10000
eps_start = 1  # eps for epsilon greedy algorithm
eps_end = 0.05
eps_anneal_period = 30
N_epochs = 1
Size_minibatch = 128
net_update_period = 1000//Size_minibatch
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
optimizer = torch.optim.RMSprop(net.parameters())

replay_buffer = ReplayBuffer(Size_replay_buffer, keys=['x', 'a', 'r', 'x_next'])
env = gym.make('CartPole-v0')
eps = eps_start

running_score_history = []

t0 = time.time()
for iter in range(N_iters):
    running_loss, running_score = 0, 0
    # collect experience
    with torch.no_grad():
        net.eval()
        s_cur = env.reset()
        s_prev = s_cur
        score = 0
        for step in range(N_steps_per_iter):
            # env.render()
            x = torch.from_numpy(np.concatenate((s_cur, s_prev))).float()
            x = x.to(device)
            if np.random.rand() < eps:
                action = np.random.randint(2)
            else:
                q = net(x.view(1, -1))
                action = np.argmax(q.detach().cpu().numpy())
            s_next, r, done, _ = env.step(action)
            if done:
                r = 0
            score += r
            x_next = torch.from_numpy(np.concatenate((s_next, s_cur))).float()
            replay_buffer.append((x, action, r, x_next))
            if done:
                running_score = 0.99*running_score + 0.01*score
                score = 0
                s_cur = env.reset()
                s_prev = s_cur
            else:
                s_prev = s_cur
                s_cur = s_next

        if eps > eps_end:
            eps -= (eps_start-eps_end)/eps_anneal_period

    # find mean and std of rewards - to make rewards zero mean and unit std for training
    # r_mean = np.mean(replay_buffer.dict['r'])
    # r_std = np.std(replay_buffer.dict['r'])
    # train
    net.train()
    for epoch in range(N_epochs):
        shuffled_ids = np.random.permutation(len(replay_buffer))
        for mb_i in range(len(replay_buffer)//Size_minibatch):
            minibatch_ids = shuffled_ids[mb_i*Size_minibatch:(mb_i+1)*Size_minibatch]
            minibatch = replay_buffer.get(minibatch_ids)
            xs, actions, rs, next_xs = minibatch.values()
            xs = torch.stack(xs).to(device)
            next_xs = torch.stack(next_xs).to(device)
            # normalize rewards
            rs = np.array(rs)
            final_state_ids = np.nonzero(rs == 0)  # will be needed to calculate target for final states properly
            # rs = (rs - r_mean)/(r_std + 1e-8)
            rs = torch.from_numpy(rs).float()

            # finding targets by double DQN method
            with torch.no_grad():
                net.eval()
                Q_snext_anext = net(next_xs)
                Q_snext_anext_ = net_(next_xs)
            net.train()
            optimizer.zero_grad()
            Q_sa = net(xs)
            Q_max, Q_argmax = torch.max(Q_snext_anext, 1)

            Q_target = torch.gather(Q_snext_anext_, 1, Q_argmax.view(-1, 1)).squeeze()
            Q_target[final_state_ids] = 0
            targets = (rs.to(device) + gamma*Q_target).to(device)
            # backprop only on actions that actually occured at corresonding states
            actions = torch.tensor(actions).view(-1, 1)
            Q_sa_relevant = torch.gather(Q_sa, 1, actions.to(device)).squeeze()
            loss = loss_function(Q_sa_relevant, targets)
            loss.backward()
            optimizer.step()
            running_loss = 0.99*running_loss + 0.01*loss.item()

            if (mb_i+1) % net_update_period == 0:
                net_.load_state_dict(net.state_dict())

    running_score_history.append(running_score)
    print("iter: {}, time: {:.2f}s, running_loss: {}, running_score: {}".format(iter, time.time()-t0,
                                                                           running_loss, running_score))
    if (iter+1) % 10 == 0:
        torch.save(net.state_dict(), net_save_path)

    if (iter+1) % 50 == 0:
        plt.close('all')
        plt.plot(running_score_history)
        plt.savefig(plot_save_path)

env.close()