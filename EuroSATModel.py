import torch
from torch import nn
import torch.nn.functional as F

# nn.BCEWithLogitsLoss

class EuroSATModel(nn.Module):

    def __init__(self, path = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.small_dropout = nn.Dropout(p=0.2)
        self.big_dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=3, out_channels=16)
        self.conv2 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=16, out_channels=16)

        self.conv3 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=16, out_channels=32)
        self.conv4 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=32, out_channels=32)

        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.fc3 = nn.Linear(in_features=1024, out_features=10)

        self.conv5 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=32, out_channels=32)
        self.conv6 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=32, out_channels=32)
        self.conv7 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=32, out_channels=32)
        self.conv8 = nn.Conv2d(kernel_size=3, padding=1, stride=1, in_channels=32, out_channels=32)

    def __name__(self):
        return "EuroSAT"

    def forward(self, x):
        # x = self.small_dropout(x)
        x = F.leaky_relu(self.conv1(x))
        # x = self.small_dropout(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        # x = self.big_dropout(x)
        x = F.leaky_relu(self.conv3(x))
        # x = self.big_dropout(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
