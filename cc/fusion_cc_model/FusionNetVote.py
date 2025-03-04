import math

from torch import nn
from torch.autograd.grad_mode import F


class FusionNetVote(nn.Module):
    def __init__(self, input_dim):
        super(FusionNetVote, self).__init__()
        self.input_dim = input_dim
        self.expert_net = ExpertNet(30)
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        # self.fc = nn.Linear(30, 10)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1024)
        self.bn1 = nn.BatchNorm1d(30)
        self.bn2 = nn.BatchNorm1d(15)
        self.fc4 = nn.Linear(1024, 30)
        self.fc5 = nn.Linear(30, 15)
        self.fc6 = nn.Linear(15, 2)

        self.dropout = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = F.relu(self.bn1(self.fc4(x)))
        x = F.relu(self.bn2(self.fc5(x)))
        x = self.bn2(self.fc6(x))
        x = self.softmax(x)
        return x

class ExpertNet(nn.Module):
    def __init__(self, input_dim):
        super(ExpertNet, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim,2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ef):
        ef = F.relu(self.bn1(self.fc1(ef)))
        ef = F.relu(self.bn2(self.fc2(ef)))
        x = self.softmax(self.fc3(ef))
        return x