import logging
from typing import List

import torch
from torch import nn

logger = logging.getLogger("cmap.ml.decoders")


class MulticlassClassification(nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x, mask):
        mask = (1 - mask.unsqueeze(-1)) * 1e6

        x = x - mask  # mask invalid timesteps
        x, _ = torch.max(x, dim=1)  # max-pooling

        x = self.linear(x.float())

        return x


class MLPDecoder(nn.Module):
    def __init__(self, sizes: List[int]):
        super().__init__()
        layers = []
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(in_features=sizes[i - 1], out_features=sizes[i]))
            layers.append(nn.BatchNorm1d(sizes[i]))
            layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)
