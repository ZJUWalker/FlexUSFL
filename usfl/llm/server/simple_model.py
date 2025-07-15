import torch
import torch.nn as nn


class SimpleServerModel(torch.nn.Module):
    def __init__(self):
        super(SimpleServerModel, self).__init__()
        self.server_blocks = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        x = self.server_blocks(x)
        return x
