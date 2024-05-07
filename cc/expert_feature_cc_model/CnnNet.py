from torch import nn


class CnnSematicNet(nn.Module):
    def __init__(self, input_dim):
        super(CnnSematicNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 64 * ((input_dim // 2) // 2)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(self.output_dim, 2)
        # self.fc1 = nn.Linear(self.output_dim, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 2)
        # self.drop_out_1 = nn.Dropout(p=0.25)
        # self.drop_out_2 = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, 1, self.input_dim)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
