from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)

        self.output = nn.Sequential(nn.Linear(64, 10), nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x= F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)       
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.output(x)
        
