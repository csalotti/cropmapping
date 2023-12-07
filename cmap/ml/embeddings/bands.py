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
        super().__init__()
        layers = []
        for i in range(1, len(sizes)):
            layers.append(SequenceLinearLayer(sizes[i-1], sizes[i]))

        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        x = x.view(batch_size*seq_length, -1)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        return x


class SequenceLinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(
                        in_features=in_features,
                        out_features=out_features,
                    )
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
