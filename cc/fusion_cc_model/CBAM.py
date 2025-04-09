import torch
from torch import nn

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