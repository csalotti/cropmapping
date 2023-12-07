import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger("lightning.pytorch.ml.embeddings.bands")


class PatchBandsEncoding(nn.Module):
    def __init__(self, channels=(32, 64, 256), kernels=(5, 5)):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=channels[0],
                kernel_size=kernels[0],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=kernels[1],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels[1]),
        )

        self.linear = nn.Linear(in_features=channels[1], out_features=channels[2])

    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)
        band_num = input_sequence.size(2)

        first_dim = batch_size * seq_length

        obs_embed = input_sequence.view(
            first_dim,
            band_num,
        ).unsqueeze(
            1
        )  # [N * T, 1, B]
        obs_embed = self.conv1(obs_embed)
        obs_embed = self.conv2(obs_embed)
        obs_embed = self.linear(
            obs_embed.view(first_dim, -1)
        )  # [batch_size*seq_length, embed_size]
        obs_embed = obs_embed.view(batch_size, seq_length, -1)

        return obs_embed


class PixelEncoding(nn.Module):
    def __init__(self, sizes: List[int]):
        self.layers = []
        for i in range(1, len(sizes)):
            self.layers.append(
                {
                    "linear": nn.Linear(
                        in_features=sizes[i - 1],
                        out_features=sizes[i],
                    ),
                    "batch_norm": nn.BatchNorm1d(sizes[i]),
                    "relu": nn.ReLU(),
                }
            )

    def forward(self, x):
        for l in self.layers:
            x = x.permute((0, 2, 1))
            x = l["linear"](x)
            x = x.permute((0, 2, 1))
            x.permute((0, 2, 1))
            x = l["batch_norm"](x)
            x = l["relu"](x)

        return x
