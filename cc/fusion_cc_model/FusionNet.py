import torch
from torch import nn
from torch.nn import functional as F
from cc.fusion_cc_model.CBAM import CBAM
from cc.fusion_cc_model.CovComponent import Encoder, MultiScaleModule
from cc.fusion_cc_model.EfComponent import ExpertNet


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.encoder = Encoder()
        self.expert_net = ExpertNet(30)
        self.fusion = ScalarWeightFusion()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, ef):
        x = self.encoder(x)
        ef = self.expert_net(ef)
        x = torch.flatten(self.pool(x), 1)
        feats = [x, ef]
        x = self.fusion(feats)
        x = self.fc(x)
        return x


class MsWithEfCovNet(nn.Module):
    def __init__(self):
        super(MsWithEfCovNet, self).__init__()
        self.ms = MultiScaleModule(128)
        self.expert_net = ExpertNet(30)
        self.pool1 = nn.MaxPool1d(2,2)
        self.conv2 = nn.Conv1d(3*128,64, 9, padding=4)
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64,2)
        self.fusion = ScalarWeightFusion()


    def forward(self, x, ef):
        x = x.reshape(-1, 1, x.shape[1])
        x = self.ms(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.flatten(self.pool2(x), 1)
        ef = self.expert_net(ef)
        feats = [x, ef]
        x = self.fusion(feats)
        x = self.fc(x)
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
