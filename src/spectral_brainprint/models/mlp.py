"""Checkpoint-compatible MLP from the team's 20-identity candidate."""

from torch import nn
from torch.nn import functional as F


class SimpleMLP(nn.Module):
    def __init__(self, n_classes, input_dim=705, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        dim = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(dim, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)])
            dim = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        h = self.body(x)
        return {"logits": self.head(h), "embedding": F.normalize(h, dim=-1)}
