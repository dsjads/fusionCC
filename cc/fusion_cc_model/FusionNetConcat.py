import math

import torch
from torch import nn
from torch.nn import functional as F


class FusionNetConcat(nn.Module):
    def __init__(self, input_dim):
        super(FusionNetConcat, self).__init__()
        self.input_dim = input_dim
        self.expert_net = ExpertNet(30)
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        # self.fc = nn.Linear(30, 10)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 64)
        self.fc4 = nn.Linear(64,30)

        self.bn1 = nn.BatchNorm1d(30)
        self.bn2 = nn.BatchNorm1d(15)
        self.fc5 = nn.Linear(60, 30)
        self.fc6 = nn.Linear(30, 15)
        self.fc7 = nn.Linear(15, 2)

        self.dropout = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, expert_feature):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        expert_feature = self.expert_net(expert_feature)
        x = torch.hstack((x,expert_feature))
        x = F.relu(self.bn1(self.fc5(x)))
        x = F.relu(self.bn2(self.fc6(x)))
        x = self.fc7(x)
        x=self.softmax(x)
        return x

class ExpertNet(nn.Module):
    def __init__(self, input_dim):
        super(ExpertNet, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, ef):
        ef = F.relu(self.bn(self.fc1(ef)))
        x = self.fc2(ef)
        return x