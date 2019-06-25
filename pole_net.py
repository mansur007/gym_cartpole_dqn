import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(4)
        self.fc1 = nn.Linear(4, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # x = self.bn1(x)
        x = F.relu(self.fc1(x))
        # x = self.bn2(x)
        x = F.relu(self.fc2(x))
        # x = self.bn3(x)
        q = self.fc3(x)
        return q

