import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMNet(nn.Module):
    def __init__(self, total_size, k=16, hidden_size=256, num_layers=2, dropout=0):
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = reshape_data(x, self.k)
        output, (h_n, c_n) = self.lstm(x)
        out = torch.concat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
        out = self.fc(out)
        out = self.softmax(out)
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
