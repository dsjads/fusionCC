import math

import torch
from torch import nn
from torch.nn import functional as F


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(32)  # 添加BN层，归一化32个通道
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(64)  # 添加BN层，归一化32个通道
        # self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.max_pool2 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        x = F.relu(self.bn1(self.conv1(x)))
        if x.size(-1) >= self.maxpool1.kernel_size:
            x = self.maxpool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class CovNetwork(nn.Module):
    def __init__(self):
        super(CovNetwork, self).__init__()
        self.encoder = Encoder()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.cbam = CBAM(64)
        self.fc = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.cbam(x)
        x = torch.flatten(self.pool(x), 1)
        x = self.softmax(self.fc(x))
        return x
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.encoder = Encoder()
        self.expert_net = ExpertNet(30)
        self.fusion = ScalarWeightFusion()
        self.cbam = CBAM(64)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, ef):
        x = self.encoder(x)
        ef = self.expert_net(ef)
        # x = self.cbam(x)
        x = torch.flatten(self.pool(x), 1)
        feats = [x, ef]
        x = self.fusion(feats)
        x = self.softmax(self.fc(x))
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        self.bn = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        if x.shape[-1] < 4:
            x = self.padding(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = self.bn(x1 + x2)
        return x

    def padding(self, x):
        padding_size = 4 - x.shape[2]
        padding = torch.zeros(x.shape[0], 1, padding_size).cuda()
        x = torch.cat([x, padding], dim=2)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModule(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModule()  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x) * x
        x = self.Sam(x) * x
        return x


class ChannelAttentionModule(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=32):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_channel, in_channel // r, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channel // r, in_channel, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):  # 空间注意力模块
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size=kernel_size, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


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
        x = self.bn2(self.fc2(ef))
        # x = self.softmax(self.fc3(x))
        return x

class ExpertCNet(nn.Module):
    def __init__(self):
        super(ExpertCNet, self).__init__()
        self.expert_net = ExpertNet(30)
        self.fc = nn.Linear(64 , 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ef):
        ef = self.expert_net(ef)
        x = self.softmax(self.fc(ef))
        return x

class ScalarWeightFusion(nn.Module):
    def __init__(self, num_branches = 2):
        super(ScalarWeightFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_branches))
        self.softmax = nn.Softmax(dim=0)
    def forward(self, feats):
        weights = self.softmax(self.weights)
        fused = sum(w*f for w, f in zip(weights,feats))
        return fused
