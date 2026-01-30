import torch
from torch import nn
import torch.nn.functional as F

# nn.BCEWithLogitsLoss

class MNISTModel(nn.Module):
    def __init__(self, path = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=1, out_channels=16)
        self.conv2 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=16, out_channels=16)
        self.conv3 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=16, out_channels=16)
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
