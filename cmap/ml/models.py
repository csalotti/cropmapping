import logging

import torch
from torch import Tensor, nn
from torch.masked import masked_tensor

logger = logging.getLogger("lightning.pytorch.ml.models")

class SITSFormerClassifier(nn.Module):
    def __init__(
        self,
        position_embedding: nn.Module,
        bands_embedding: nn.Module,
        transformer_encoder: nn.Module,
        classification_head: nn.Module,
        dropout_p: float = 0.1,
    ):
        nn.Module.__init__(self)

        self.position_embedding = position_embedding
        self.bands_embedding = bands_embedding
        self.transformer_encoder = transformer_encoder
        self.classification_head = classification_head
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, ts: Tensor, days: Tensor, mask: Tensor):
        logger.debug(f"ts : {ts}\ndays : {days}\nmask : {mask}")
        x_position_emb = self.position_embedding(days)
        x_bands_emb = self.bands_embedding(ts)
        logger.debug(f"position : {x_position_emb}\nbands : {x_bands_emb}")

        x_emb = self.dropout(x_position_emb + x_bands_emb)
        logger.debug(f"post dropout : {x_emb}")

        # Transpose N and T for convenience in masking
        x_trans = self.transformer_encoder(x_emb, src_key_padding_mask=mask)
        logger.debug(f"Transformer: {x_trans}")

        return self.classification_head(x_trans, mask)


class MulticlassClassification(nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x, mask):
        mask = (~mask.unsqueeze(-1)).float() * 1e6
        x = x - mask  # mask invalid timesteps
        x, _ = torch.max(x, dim=1)  # max-pooling
        logger.debug(f"After Max : {x}")
        return self.linear(x)
