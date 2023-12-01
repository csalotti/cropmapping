import logging

import torch
import torch.nn as nn

logger = logging.getLogger("lightning.pytorch.ml.embeddings.bands")


class PatchBandsEncoding(nn.Module):
    def __init__(self, channel_size=(32, 64, 256), kernel_sizes=(5, 5)):
        super().__init__()

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=1,
        #         out_channels=channel_size[0],
        #         kernel_size=kernel_sizes[0],
        #     ),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(channel_size[0]),
        # )
        #
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=channel_size[0],
        #         out_channels=channel_size[1],
        #         kernel_size=kernel_sizes[1],
        #     ),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(channel_size[1]),
        # )
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=channel_size[0],
                kernel_size=(kernel_sizes[0], 1, 1),
            ),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[0]),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=channel_size[0],
                out_channels=channel_size[1],
                kernel_size=(kernel_sizes[1], 1, 1),
            ),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[1]),
        )

        self.linear = nn.Linear(
            in_features=2 * channel_size[1], out_features=channel_size[2]
        )

        self.embed_size = channel_size[-1]

    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)
        band_num = input_sequence.size(2)
        patch_size = 1
        first_dim = batch_size * seq_length

        obs_embed = input_sequence.view(
            first_dim, band_num, patch_size, patch_size
        ).unsqueeze(1)
        obs_embed = self.conv1(obs_embed)
        obs_embed = self.conv2(obs_embed)
        obs_embed = self.linear(
            obs_embed.view(first_dim, -1)
        )  # [batch_size*seq_length, embed_size]
        obs_embed = obs_embed.view(batch_size, seq_length, -1)

        return obs_embed
