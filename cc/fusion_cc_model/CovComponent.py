import torch
from torch import nn
from torch.nn import functional as F
from cc.fusion_cc_model.CBAM import CBAM


class MsCovNet(nn.Module):
    def __init__(self):
        super(MsCovNet, self).__init__()
        self.ms = MultiScaleModule(128)
        self.pool1 = nn.MaxPool1d(2,2)
        self.conv2 = nn.Conv1d(3*128,64, 9, padding=4)
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64,2)


    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        x = self.ms(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.flatten(self.pool2(x), 1)
        x = self.fc(x)
        return x

class MultiScaleModule(nn.Module):
    def __init__(self, output_channel = 128):
        super(MultiScaleModule, self).__init__()
        self.conv1 = nn.Conv1d(1, output_channel, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(1, output_channel, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(1, output_channel, kernel_size=9, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.concat([x1, x2, x3], dim=1)
        return x

class CovNetwork(nn.Module):
    def __init__(self):
        super(CovNetwork, self).__init__()
        self.encoder = Encoder()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.cbam = CBAM(64)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.cbam(x)
        x = torch.flatten(self.pool(x), 1)
        x = self.fc(x)
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

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        if x.shape[-1] < 4:
            x = self.padding(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = x1 + x2
        return x

    def padding(self, x):
        padding_size = 4 - x.shape[2]
        padding = torch.zeros(x.shape[0], 1, padding_size).cuda()
        x = torch.cat([x, padding], dim=2)
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


