import math

from torch import nn
from torch.nn import functional as F


class MlpSematicNet(nn.Module):
    def __init__(self, input_dim):
        super(MlpSematicNet, self).__init__()
        self.input_dim = input_dim
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = F.relu(self.bn1(self.fc4(x)))
        x = F.relu(self.bn2(self.fc5(x)))
        x = self.fc6(x)
        x = self.softmax(x)
        return x
