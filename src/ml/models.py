from torch import Tensor, nn


class SITSFormerClassifier(nn.Module):
    def __init__(
        self,
        position_embedding: nn.Module,
        bands_embedding: nn.Module,
        transformer_encoder: nn.Module,
        classification_head: nn.Module,
        dropout_p: float = 0.1,
    ):
        self.position_embedding = position_embedding
        self.bands_embedding = bands_embedding
        self.transformer_encoder = transformer_encoder
        self.classification_head = classification_head
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, ts: Tensor, days: Tensor, mask: Tensor):
        x_position_emb = self.position_embedding(days)
        x_bands_emb = self.position_embedding(ts)

        x_emb = self.dropout(x_position_emb + x_bands_emb)

        x_trans = self.transformer_encoder(x_emb, src_key_padding_mask=mask)

        return self.classification_head(x_trans, mask)
