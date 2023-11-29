import logging
from typing import List

from torch import Tensor, nn
import torch

from cmap.ml.embeddings.bands import PatchBandsEncoding
from cmap.ml.embeddings.position import PositionalEncoding
from cmap.utils.attention import SaveAttentionMapHook, patch_attention

logger = logging.getLogger("cmap.ml.encoders")


class SITSFormer(nn.Module):
    def __init__(
        self,
        max_n_days: int = 397,
        d_model: int = 256,
        band_emb_chanels: List[int] = [32, 64],
        band_emb_kernel_size: List[int] = [5, 5],
        att_hidden_size: int = 256,
        n_att_head: int = 8,
        n_att_layers: int = 3,
        dropout_p: float = 0.1,
    ):
        nn.Module.__init__(self)

        self.position_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_n_days,
        )
        self.bands_encoder = PatchBandsEncoding(
            channel_size=band_emb_chanels + [d_model],
            kernel_sizes=band_emb_kernel_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_att_head,
            dim_feedforward=att_hidden_size * 4,
            dropout=dropout_p,
        )
        encoder_norm = nn.LayerNorm(att_hidden_size)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_att_layers,
            norm=encoder_norm,
        )

        self.dropout = nn.Dropout(dropout_p)


    @torch.no_grad()
    def get_attention_maps(self, ts: Tensor, days: Tensor, mask: Tensor):
        outpout_hook = SaveAttentionMapHook()
        hooks_handles = [
            l.self_attn.register_forward_hook(outpout_hook)
            for l in self.transformer_encoder.layers
        ]
        self.forward(ts=ts, mask=mask, days=days)
        for h in hooks_handles:
            h.remove()

        outputs = outpout_hook.outputs
        # Merge layers on dim 1
        outputs = torch.concatenate([o.unsqueeze(1) for o in outputs], dim=1)

        return outputs

    def forward(self, ts: Tensor, days: Tensor, mask: Tensor):
        logger.debug(f"ts : {ts}\ndays : {days}\nmask : {mask}")
        x_position_emb = self.position_encoder(days)
        x_bands_emb = self.bands_encoder(ts)
        logger.debug(f"position : {x_position_emb}\nbands : {x_bands_emb}")

        x_emb = self.dropout(x_position_emb + x_bands_emb)
        logger.debug(f"post dropout : {x_emb}")

        # Transpose N and T for convenience in masking
        x_emb = x_emb.transpose(0, 1)
        x_trans = self.transformer_encoder(x_emb, src_key_padding_mask=~mask)
        x_trans = x_trans.transpose(0, 1)
        logger.debug(f"Transformer: {x_trans}")

        return x_trans
