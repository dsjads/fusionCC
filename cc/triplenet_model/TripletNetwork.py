import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.drop_out_1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.drop_out_2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.drop_out_3 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(hidden_dim_1, input_dim)
        self.drop_out_4 = nn.Dropout(p=0.25)
        self.fc5 = nn.Linear(input_dim, 20)

    def forward(self, x):
        # x = self.fc1(x)
        temp_x = self.fc1(x)
        temp_x = self.drop_out_1(temp_x)
        temp_x = self.fc2(temp_x)
        temp_x = self.drop_out_2(temp_x)
        temp_x = self.fc3(temp_x)
        temp_x = self.drop_out_3(temp_x)
        temp_x = self.fc4(temp_x)
        temp_x = self.drop_out_4(temp_x)
        x = self.fc5(temp_x)
        return x


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, y, z):
        # reference
        embedded_x = self.embeddingnet(x)
        # passing
        embedded_y = self.embeddingnet(y)
        # failing
        embedded_z = self.embeddingnet(z)

        dist_a = F.pairwise_distance(embedded_x, embedded_y, p=2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, p=2)
        a = torch.vstack((dist_a, dist_b))
        prob = self.softmax(a)
        return dist_a, dist_b, prob, embedded_x, embedded_y, embedded_z
