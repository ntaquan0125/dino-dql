import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights


class BasicCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, out_channels)

    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.view(B, L * C, H, W)
        x = self.feature(x)
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicLSTM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.lstm = nn.LSTM(4096, 1024, 1, batch_first=True)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.view(B * L, C, H, W)
        x = self.feature(x)
        x = x.view(B, L, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x