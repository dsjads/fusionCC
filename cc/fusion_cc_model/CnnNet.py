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