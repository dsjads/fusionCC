import math
from torch import nn
from torch.nn import functional as F

class ExpertNet(nn.Module):
    def __init__(self, input_dim):
        super(ExpertNet, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(64)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, ef):
        ef = F.relu(self.bn1(self.fc1(ef)))
        x = self.fc2(ef)
        return x

class ExpertCNet(nn.Module):
    def __init__(self):
        super(ExpertCNet, self).__init__()
        self.expert_net = ExpertNet(30)
        self.fc = nn.Linear(64 , 2)

    def forward(self, ef):
        ef = self.expert_net(ef)
        x = self.fc(ef)
        return x