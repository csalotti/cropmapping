import logging
from typing import List

from torch import Tensor, dropout, nn
import torch

from cmap.ml.embeddings.bands import PatchBandsEncoding
from cmap.ml.embeddings.position import PositionalEncoding
from cmap.utils.attention import SaveAttentionMapHook, patch_attention

logger = logging.getLogger("cmap.ml.encoders")


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        position_encoder: nn.Module,
        bands_encoder: nn.Module,
        d_model: int = 256,
        n_att_head: int = 8,
        n_att_layers: int = 3,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.position_encoder = position_encoder
        self.bands_encoder = bands_encoder

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_att_head,
            dim_feedforward=d_model * 4,
            dropout=dropout_p,
        )
        encoder_norm = nn.LayerNorm(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_att_layers,
            norm=encoder_norm,
        )

        self.dropout = nn.Dropout(dropout_p)

    @torch.no_grad()
    def get_attention_maps(self, ts: Tensor, positions: Tensor, mask: Tensor):
        outpout_hook = SaveAttentionMapHook()
        hooks_handles = [
            l.self_attn.register_forward_hook(outpout_hook)
            for l in self.transformer_encoder.layers
        ]
        self.forward(ts=ts, mask=mask, positions=positions)
        for h in hooks_handles:
            h.remove()

        outputs = outpout_hook.outputs
        # Merge layers on dim 1
        outputs = torch.concatenate(
            [o.unsqueeze(1) for o in outputs], dim=1
        )  # [N, L, S, S]

        return outputs

    def forward(self, ts: Tensor, positions: Tensor, mask: Tensor):
        mask = mask == 0

        x_position_emb = self.position_encoder(positions)
        x_bands_emb = self.bands_encoder(ts)

        x_emb = self.dropout(x_position_emb + x_bands_emb)

        # Transpose N and T for convenience in masking
        x_emb = x_emb.transpose(0, 1)
        x_trans = self.transformer_encoder(x_emb, src_key_padding_mask=mask)
        x_trans = x_trans.transpose(0, 1)

        return x_trans


class SITSFormer(TransformerEncoder):
    def __init__(
        self,
        max_n_days: int = 397,
        d_model: int = 256,
        band_emb_chanels: List[int] = [32, 64],
        band_emb_kernel_size: List[int] = [5, 5],
        n_att_head: int = 8,
        n_att_layers: int = 3,
        dropout_p: float = 0.1,
    ):
        position_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_n_days,
        )
        bands_encoder = PatchBandsEncoding(
            channel_size=band_emb_chanels + [d_model],
            kernel_sizes=band_emb_kernel_size,
        )

        super().__init__(
            position_encoder,
            bands_encoder,
            d_model,
            n_att_head,
            n_att_layers,
            dropout_p,
        )
