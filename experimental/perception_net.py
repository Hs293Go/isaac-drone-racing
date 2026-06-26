"""Shared perception net: FPV image + proprio -> predicted 20-dim gate-relative state.

Imported by perception_train.py (trains it) and perception_eval.py (composes it with the
state control policy). Pure torch — no Isaac imports — so it loads without the app boot.
"""

import torch
from torch import nn

CAM_RES = 64
PROPRIO_DIM = 13
IMG_DIM = CAM_RES * CAM_RES * 3


class PerceptionNet(nn.Module):
    """FPV image + proprio -> predicted 20-dim gate-relative state (the racer's obs)."""

    def __init__(self, out_dim: int = 20):
        """CNN over the image, concatenated with proprio, -> out_dim state values."""
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )  # 64x64 -> 1024
        self.head = nn.Sequential(
            nn.Linear(1024 + PROPRIO_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, vis):
        """Flat [image | proprio] obs -> predicted state."""
        img = vis[:, :IMG_DIM].reshape(-1, CAM_RES, CAM_RES, 3).permute(0, 3, 1, 2)
        return self.head(torch.cat([self.cnn(img), vis[:, IMG_DIM:]], dim=-1))
