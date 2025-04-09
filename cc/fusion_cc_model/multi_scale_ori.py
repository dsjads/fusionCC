from torch import nn
import torch
from torch.nn import functional as F
from cc.fusion_cc_model.EfComponent import ExpertNet
from cc.fusion_cc_model.FusionNet import ScalarWeightFusion

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


class MSResNet(nn.Module):
    def __init__(self):
        super(MSResNet, self).__init__()
        self.relu = nn.ReLU()
        self.layer3 = BasicBlock(1, 64, 3)
        self.layer5 = BasicBlock(1, 64, 5)
        self.layer7 = BasicBlock(1, 64, 7)
        self.conv = nn.Conv1d(3 * 64, 64, 9, padding=4)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.pool2 = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(64, 2)
    def forward(self, x0):
        x0 = x0.unsqueeze(1)
        x = self.layer3(x0)
        y = self.layer5(x0)
        z = self.layer7(x0)
        out = torch.cat([x,y,z],dim=1)
        out = self.pool1(out)
        out = self.conv(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class MSFusionNet(nn.Module):
    def __init__(self):
        super(MSFusionNet, self).__init__()
        self.relu = nn.ReLU()
        self.layer3 = BasicBlock(1, 64, 3)
        self.layer5 = BasicBlock(1, 64, 5)
        self.layer7 = BasicBlock(1, 64, 7)
        self.conv = nn.Conv1d(3 * 64, 64, 9, padding=4)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.handcraftedNet = ExpertNet(30)
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
            nn.Conv1d(1,64,7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer2 = BasicBlock(
            nn.Conv1d(64, 128, 7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(128, 128, 7, stride=1, padding=3),
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