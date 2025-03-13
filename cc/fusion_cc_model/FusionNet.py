import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fusioning(enginnered, values):
    fused_tensor = enginnered.unsqueeze(1) + values.unsqueeze(2)
    return fused_tensor.permute(0, 2, 1)


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.expert_net = ExpertNet(30)
        self.cov_net = CnnNet()
        self.fc = nn.Linear(64, 30)
        self.fc1 = nn.Linear(60, 32)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, ef):
        x = self.fc(self.cov_net(x))
        ef = self.expert_net(ef)
        x = torch.hstack((x, ef))
        x = F.relu(self.bn(self.fc1(x)))
        x = self.softmax(self.fc2(x))
        return x


class FusionNet1(nn.Module):
    def __init__(self, input_dim):
        super(FusionNet1, self).__init__()
        self.input_dim = input_dim
        self.expert_net = ExpertNet(30)
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        # self.fc = nn.Linear(30, 10)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1024)

        self.conv1 = torch.nn.Conv1d(30, 64, 1)
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

    def forward(self, x, expert_feature):
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        expert_feature = self.expert_net(expert_feature)

        x = fusioning(expert_feature, x)
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


class ExpertNet(nn.Module):
    def __init__(self, input_dim):
        super(ExpertNet, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        # self.fc3 = nn.Linear(input_dim,2)
        self.bn = nn.BatchNorm1d(hidden_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, ef):
        ef = F.relu(self.bn(self.fc1(ef)))
        x = F.relu(self.fc2(ef))
        # x = self.softmax(self.fc3(x))
        return x


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(32)  # 添加BN层，归一化32个通道
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(64)  # 添加BN层，归一化32个通道
        # self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveMaxPool1d(1)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        x = F.relu(self.bn1(self.conv1(x)))
        if x.size(-1) >= self.maxpool1.kernel_size:
            x = self.maxpool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avg_pool(x)
        x = self.bn3(x.view(x.size(0), -1))
        return x


class SupCENet(nn.Module):
    def __init__(self, dim, encoder):
        super(SupCENet, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(dim, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class ExpertNet2(nn.Module):
    def __init__(self, input_dim):
        super(ExpertNet2, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, 2)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ef):
        ef = F.relu(self.bn(self.fc1(ef)))
        x = F.relu(self.fc2(ef))
        x = self.softmax(self.fc3(x))
        return x


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ECAttention(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class PSAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.collect = nn.Conv2d(out_channels, out_channels, 1)
        self.distribute = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv_reduce(x)
        b, c, h, w = x.size()
        # Collect
        x_collect = self.collect(x).view(b, c, -1)
        x_collect = F.softmax(x_collect, dim=-1)
        # Distribute
        x_distribute = self.distribute(x).view(b, c, -1)
        x_distribute = F.softmax(x_distribute, dim=1)
        # Attention
        x_att = torch.bmm(x_collect, x_distribute.permute(0, 2, 1)).view(b, c, h, w)
        return x + x_att

