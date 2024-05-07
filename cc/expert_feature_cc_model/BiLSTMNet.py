import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMNet(nn.Module):
    def __init__(self, k, total_size, num_layers, dropout):
        super(BiLSTMNet, self).__init__()
        self.k = k
        self.input_size = math.ceil(total_size / k)
        self.hidden_size = math.floor(self.input_size * 1.5)
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=self.dropout)

        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = reshape_data(x, self.k)
        output, (h_n, c_n) = self.lstm(x)
        out = torch.concat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
        out = F.relu(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


def reshape_data(data, k):
    batch_size, n_features = data.size()
    padding_needed = (k - (n_features % k)) % k

    padding_shape = (batch_size, padding_needed)
    zeros = torch.zeros(*padding_shape, dtype=data.dtype, device=data.device)
    padded_data = torch.cat((data, zeros), dim=1)

    seq_len = padded_data.size(1) // k
    reshaped_data = padded_data.view(batch_size, seq_len, k)
    reshaped_data = reshaped_data.permute(0, 2, 1)
    return reshaped_data
