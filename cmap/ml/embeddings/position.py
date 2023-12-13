import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positionnal embedding, based on the original paper Attention
    is all you need

    Attributes:
        encoding (nn.Embedding) : Sin/Cos Look-up table
            of shape [max_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        tau: float = 10000.0,
        max_len: int = 397,
    ):
        """PositionalEncoding initialization

        Args:
            d_model (int) : transformer encoder latent size
            tau (float) : log space size
            max_len (int) : eemmbedding size

        """
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len + 1, d_model)

        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(tau) / d_model)
        )  # [d_model/2,]

        # keep pe[0,:] to zeros
        pe[1:, 0::2] = torch.sin(
            position * div_term
        )  # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(
            position * div_term
        )  # broadcasting to [max_len, d_model/2]

        self.encoding = nn.Embedding.from_pretrained(pe, freeze=True)

    def forward(self, positions):
        return self.encoding(positions)  # [batch_size, seq_length, embed_dim]
