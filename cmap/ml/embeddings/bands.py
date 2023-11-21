import torch.nn as nn


class PatchBandsEncoding(nn.Module):
    def __init__(self, channel_size=(32, 64, 256), kernel_size=(5, 1, 5, 1)):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=channel_size[0],
                kernel_size=(kernel_size[0], kernel_size[1], kernel_size[1]),
            ),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[0]),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=channel_size[0],
                out_channels=channel_size[1],
                kernel_size=(kernel_size[2], kernel_size[3], kernel_size[3]),
            ),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[1]),
        )

        self.linear = nn.Linear(
            in_features=channel_size[1], out_features=channel_size[2]
        )

        self.embed_size = channel_size[-1]

    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)
        band_num = input_sequence.size(2)
        patch_size = input_sequence.size(3)
        first_dim = batch_size * seq_length

        obs_embed = input_sequence.view(
            first_dim,
            band_num,
            patch_size,
            patch_size,
        ).unsqueeze(
            1
        )  # [B * T, 1, B, 1, 1]
        obs_embed = self.conv1(obs_embed)
        obs_embed = self.conv2(obs_embed)
        obs_embed = self.linear(
            obs_embed.view(first_dim, -1)
        )  # [batch_size*seq_length, embed_size]
        obs_embed = obs_embed.view(batch_size, seq_length, -1)

        return obs_embed
