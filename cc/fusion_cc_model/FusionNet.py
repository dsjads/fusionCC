import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fusioning(x1, x2):
    # expert = expert.unsqueeze(2).expand(-1, -1, values.size(1))
    batch_size, dim1 = x1.shape
    _, dim2 = x2.shape
    fused_tensor = torch.zeros((batch_size, dim1 + 1, dim2 + 1), device=x1.device)

    # 计算外积
    x1_expanded = x1.unsqueeze(2)  # (batch_size, dim1, 1)
    x2_expanded = x2.unsqueeze(1)  # (batch_size, 1, dim2)
    fused_tensor[:, 1:, 1:] = x1_expanded * x2_expanded  # (batch_size, dim1, dim2)

    # 将 x1 拼接到第一列
    fused_tensor[:, 1:, 0] = x1

    # 将 x2 拼接到第一行
    fused_tensor[:, 0, 1:] = x2

    # 第一行第一列的位置设置为0（默认已经为0）

    return fused_tensor
    # repeat_values = values.unsqueeze(2)
    # repeat_values = repeat_values.repeat(1, 1, expert.size(1))
    # weighted_matrix = expert + repeat_values.permute(0,2,1)
    # return weighted_matrix.transpose(2, 1)
    # weighted_matrix = expert.permute(0, 2, 1) * values.unsqueeze(2)
    # return fused_tensor


class FusionNet(nn.Module):
    def __init__(self, input_dim):
        super(FusionNet, self).__init__()
        self.input_dim = input_dim
        self.expert_net = ExpertNet(30)
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        # self.fc = nn.Linear(30, 10)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1024)

        self.conv1 = torch.nn.Conv1d(31, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(1025, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, expert_feature):
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        expert_feature = self.expert_net(expert_feature)

        x = fusioning(expert_feature, x)
        # n_stats = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.transpose(2, 1)
        x = torch.sum(x, dim=2, keepdim=True)
        x = x.view(-1, 1025)

        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        x = self.softmax(x)
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

class ExpertNet1(nn.Module):
    def __init__(self, input_dim):
        super(ExpertNet1, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim,2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, ef):
        ef = F.relu(self.bn1(self.fc1(ef)))
        ef = F.relu(self.bn2(self.fc2(ef)))
        x = self.fc3(ef)
        return x

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # x = self.fc2(self.relu(self.fc1(x)))
        x = F.relu(self.bn(self.fc1(x)))
        x = self.fc2(x)
        return x


class FusionNet1(nn.Module):
    def __init__(self, input_dim):
        super(FusionNet1, self).__init__()
        self.input_dim = input_dim
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1024)

        self.conv1 = torch.nn.Conv1d(1, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.transpose(2, 1)
        x = torch.sum(x, dim=2, keepdim=True)
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        x = self.softmax(x)
        return x


class FusionNet2(nn.Module):
    def __init__(self, input_dim):
        super(FusionNet2, self).__init__()
        self.input_dim = input_dim
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.fc4 = nn.Linear(hidden_dim_1, input_dim)
        self.fc5 = nn.Linear(input_dim, 2)
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(hidden_dim_2)

    def forward(self, x, expert_feature):
        return x
