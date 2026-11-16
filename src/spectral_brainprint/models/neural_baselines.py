"""Residual MLP and raw EEG CNN from the Brainprint_ML grouped comparisons."""

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(w),
            nn.Linear(w, w * 2),
            nn.GELU(),
            nn.Dropout(.2),
            nn.Linear(w * 2, w),
            nn.Dropout(.2),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    """Process normalized team feature vectors with shape (batch, 705)."""

    def __init__(self, n):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(705, 256),
            nn.GELU(),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.LayerNorm(256),
        )
        self.head = nn.Linear(256, n)

    def forward(self, x):
        features = self.body(x)
        return {'logits': self.head(features), 'embedding': F.normalize(features, dim=-1)}


class RawCNN(nn.Module):
    """Classify normalized EEG tensors of shape (batch, 1, 18, 1200)."""

    def __init__(self, n):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 8, (1, 151), padding=(0, 75), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (18, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(.5),
            nn.Conv2d(16, 16, (1, 31), padding=(0, 15), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(.5),
            nn.Flatten(),
        )
        self.head = nn.Linear(16 * 37, n)

    def forward(self, x):
        features = self.body(x)
        return {'logits': self.head(features), 'embedding': F.normalize(features, dim=-1)}
