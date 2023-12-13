import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger("lightning.pytorch.ml.embeddings.bands")


class PatchBandsEncoding(nn.Module):
    """Convolution based encoding for bands. Based on SITS former, it
    originally encodes patches in space and time dimensions with 3D convolutions,
    flatten the features map and pass through a final projection.

    Attributes:
        conv1 (nn.Sequential) : First convolution Layer
        conv2 (nn.Sequential) : Second convolution Layer
        linear (nn.Linear) : Last projection layer

    """

    def __init__(self, channels=(32, 64, 256), kernels=(5, 5)):
        """PatchBandsEncoding init

        Args:
            channels (Tuple(int, int, int)) : Features map size
            kernels (Tuple(int, int)) : convolutions kernels sizes

        """
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

    def forward(self, ts):
        """PatchBandsEncoding forward pass. Applied only on bands,
        no reduction or augmentations on sequence size is done.

        Notes on shapes:
            N : Batch size
            S : Sequence size
            B : Number of bandsk
            E : Final band embeddings size

        Args:
            ts (Tensor) : Input sequence of shape [N,S,B]

        Returns:
            Encoded sequence of shape [N, S, E]
        """
        batch_size = ts.size(0)
        seq_length = ts.size(1)
        band_num = ts.size(2)

        first_dim = batch_size * seq_length

        # Reshape to [N * S, 1, B] for applying
        # convolutions only on bands
        ts = ts.view(
            first_dim,
            band_num,
        ).unsqueeze(1)
        ts_embed = self.conv1(ts)
        ts_embed = self.conv2(ts_embed)
        ts_embed = self.linear(ts_embed.view(first_dim, -1))
        # Shape [N, S, E]
        ts_embed = ts_embed.view(batch_size, seq_length, -1)

        return ts_embed


class PixelEncoding(nn.Module):
    """MLP on pixel level time series. It consists of a succession
    of bands non-linear tranasformations and combinations.

    Attributes:
        encoder (nn.Sequential) : MLP encoder (Linear + BatchNorm1d + ReLU)

    """

    def __init__(self, sizes: List[int]):
        """PixelEncoding initialization

        Args:
            sizes (List[int]) : MLP hidden layers sizes
        """
        super().__init__()
        layers = []
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(in_features=sizes[i - 1], out_features=sizes[i]))
            layers.append(nn.BatchNorm1d(sizes[i]))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, ts):
        """PixelBandsEncoding forward pass. Applied only on bands,
        no reduction or augmentations on sequence size is done.

        Notes on shapes:
            N : Batch size
            S : Sequence size
            B : Number of bands
            E : Final band embeddings size

        Args:
            ts (Tensor) : Input sequence of shape [N,S,B]

        Returns:
            Encoded sequence of shape [N, S, E]
        """
        batch_size = ts.size(0)
        seq_length = ts.size(1)
        ts = ts.view(batch_size * seq_length, -1)
        ts_encoded = self.encoder(ts)
        ts_encoded = ts_encoded.view(batch_size, seq_length, -1)
        return ts_encoded
