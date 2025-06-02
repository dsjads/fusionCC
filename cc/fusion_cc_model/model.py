import math

from torch import nn
import torch
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(BasicBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        # downsample
        if in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1),
                nn.BatchNorm1d(out_channel)
            )
        else:
            self.downsample = None
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = residual + out
        out = self.relu(out)
        return out


class MSFusionNet(nn.Module):
    def __init__(self):
        super(MSFusionNet, self).__init__()
        self.relu = nn.ReLU()
        self.layer3 = BasicBlock(1, 64, 3)
        self.layer5 = BasicBlock(1, 64, 5)
        self.layer7 = BasicBlock(1, 64, 9)
        self.conv = nn.Conv1d(3 * 64, 64, 9, padding=4)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.handcraftedNet = HandcraftedNet(30)
        self.fusion = ScalarWeightFusion()
        self.fc = nn.Linear(64, 2)
    def forward(self, x0 , ef):
        x0 = x0.unsqueeze(1)
        x = self.layer3(x0)
        y = self.layer5(x0)
        z = self.layer7(x0)
        out = torch.cat([x, y, z], dim=1)
        out = self.pool1(out)
        out = self.conv(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        ef = self.handcraftedNet(ef)
        feats = [out,ef]
        x = self.fusion(feats)
        x = self.fc(x)
        return x


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,64,9, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,9, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer2 = BasicBlock(
            nn.Conv1d(64, 128, 9, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(128, 128, 9, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2, 2)
        self.pool2 = nn.AdaptiveMaxPool1d(1)

    def forward(self, x0):
        x0 = x0.unsqueeze(1)
        x = self.layer1(x0)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        out = torch.flatten(x, 1)
        out = self.fc(out)
        return out


class ScalarWeightFusion(nn.Module):
    def __init__(self, num_branches = 2):
        super(ScalarWeightFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_branches))
        self.softmax = nn.Softmax(dim=0)
    def forward(self, feats):
        weights = self.softmax(self.weights)
        fused = sum(w*f for w, f in zip(weights,feats))
        return fused


class HandcraftedNet(nn.Module):
    def __init__(self, input_dim):
        super(HandcraftedNet, self).__init__()
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

class HandcraftedCNet(nn.Module):
    def __init__(self):
        super(HandcraftedCNet, self).__init__()
        self.expert_net = HandcraftedNet(30)
        self.fc = nn.Linear(64 , 2)

    def forward(self, ef):
        ef = self.expert_net(ef)
        x = self.fc(ef)
        return x
