import math

from torch import nn
from torch.nn import functional as F


class CnnSematicNet(nn.Module):
    def __init__(self, input_dim):
        super(CnnSematicNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 64 if input_dim // 16 == 0 else 64 * (input_dim // 16)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4)
        padding_needed1 = max(0, math.ceil((4 - input_dim) / 2))
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=padding_needed1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        padding_needed2 = max(0, math.ceil((4 - input_dim // 4) / 2))
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=padding_needed2)
        self.fc1 = nn.Linear(self.output_dim, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, 1, self.input_dim)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.softmax(self.fc2(x))
        return x
