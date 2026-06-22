import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNArchitecture(nn.Module):
    def __init__(self):
        super(CNNArchitecture, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = CNNArchitecture()
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)

    print(model)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)