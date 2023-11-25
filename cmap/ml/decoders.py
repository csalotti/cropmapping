import logging

import torch
from torch import nn

logger = logging.getLogger("cmap.ml.decoders")


class MulticlassClassification(nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x, mask):
        mask = (~mask.unsqueeze(-1)).float() * 1e6
        x = x - mask  # mask invalid timesteps
        x, _ = torch.max(x, dim=1)  # max-pooling
        return self.linear(x)
