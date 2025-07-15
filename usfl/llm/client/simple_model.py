import os

import torch
import torch.nn as nn

# 拆分学习，服务端， 模型头部


class SimpleHeadModel(torch.nn.Module):
    def __init__(self):
        super(SimpleHeadModel, self).__init__()
        self.embedding = nn.Embedding(100, 64)
        self.head_blocks = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.head_blocks(x)
        return x


class SimpleTailModel(torch.nn.Module):
    def __init__(self):
        super(SimpleTailModel, self).__init__()
        self.tail_blocks = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64, 100, bias=False)  # 全连接层
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        x = self.tail_blocks(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
