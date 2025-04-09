import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMNet(nn.Module):
    def __init__(self, k=16, hidden_size=256, num_layers=2, dropout=0):
        super(BiLSTMNet, self).__init__()
        self.k = k
        self.input_size = math.ceil(k)
        self.hidden_size = math.floor(hidden_size)
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=self.dropout)

        self.fc = nn.Linear(self.hidden_size * 2, 2)

    def forward(self, x):
        x = reshape_data(x, self.k)
        output, (h_n, c_n) = self.lstm(x)
        out = torch.concat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
        out = self.fc(out)
        return out


def reshape_data(data, k):
    batch_size, n_features = data.size()
    # 需要填充的维度
    padding_needed = (k - (n_features % k)) % k
    padding_shape = (batch_size, padding_needed)
    # 填充的零矩阵
    zeros = torch.zeros(*padding_shape, dtype=data.dtype, device=data.device)
    # 填充
    padded_data = torch.cat((data, zeros), dim=1)
    seq_len = padded_data.size(1) // k
    reshaped_data = padded_data.view(batch_size, seq_len, k)
    reshaped_data = reshaped_data.permute(0, 1, 2)
    return reshaped_data

class CnnNet(nn.Module):
    def __init__(self, input_dim):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 3*64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(3*64, 64, kernel_size=9, stride=1, padding=4)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        if x.shape[-1] < 4:
            x = self.padding(x)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

    def padding(self, x):
        padding_size = 4 - x.shape[2]
        padding = torch.zeros(x.shape[0], 1, padding_size).cuda()
        x = torch.cat([x, padding], dim=2)
        return x


class MlpNet(nn.Module):
    def __init__(self, input_dim):
        super(MlpNet, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.fc4 = nn.Linear(input_dim, 2)


    def forward(self, x):
        # x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
