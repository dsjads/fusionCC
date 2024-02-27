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


class Net1(nn.Module):
    def __init__(self, input_dim):
        super(Net1, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        # x = self.fc2(self.relu(self.fc1(x)))
        return x


class Net2(nn.Module):
    def __init__(self, input_dim):
        super(Net2, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.fc2(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        return x


class Net3(nn.Module):
    def __init__(self, input_dim):
        super(Net3, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        # x = self.fc2(self.relu(self.fc1(x)))
        return x


class Net4(nn.Module):
    def __init__(self, input_dim, net1, net2, net3):
        super(Net4, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3
        self.col = input_dim // 3
        self.input_dim = input_dim
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.fc4 = nn.Linear(hidden_dim_1, input_dim)
        self.fc5 = nn.Linear(input_dim, 10)

    def forward(self, x):
        n = self.col
        ssp = x[:, :n]
        cr = x[:, n:2 * n]
        sf = x[:, 2 * n: 3 * n]

        ssp = self.net1(ssp)
        cr = self.net2(cr)
        sf = self.net3(sf)
        x = torch.cat((ssp, cr, sf), dim=1)
        x = self.fc5(self.relu(self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))))
        # x=self.fc5(x)
        return x


class TripletWithExpertFeatureNet(nn.Module):
    def __init__(self, coverage_info_net, expert_feature_net):
        super(TripletWithExpertFeatureNet, self).__init__()
        self.coverage_info_net = coverage_info_net

        self.expert_feature_net = expert_feature_net
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x, x_feature, y, y_feature, z, z_feature):
        # reference
        embedded_x = self.coverage_info_net(x)
        # passing
        embedded_y = self.coverage_info_net(y)
        # failing
        embedded_z = self.coverage_info_net(z)

        x_feature = self.expert_feature_net(x_feature)
        y_feature = self.expert_feature_net(y_feature)
        z_feature = self.expert_feature_net(z_feature)
        embedded_x = torch.cat((embedded_x, x_feature), dim=1)
        embedded_y = torch.cat((embedded_y, y_feature), dim=1)
        embedded_z = torch.cat((embedded_z, z_feature), dim=1)

        dist_a = F.pairwise_distance(embedded_x, embedded_y, p=2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, p=2)
        a = torch.vstack((dist_a, dist_b))
        prob = self.softmax(a)
        return dist_a, dist_b, prob, embedded_x, embedded_y, embedded_z
