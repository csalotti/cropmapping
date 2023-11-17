from typing import List, Tuple
import pytorch_lightning as L
from pytorch_lightning.callbacks import lr_finder
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch.nn as nn
from torch.optim import Adam
from embeddings.position import PositionalEncoding
from embeddings.bands import CNN3DEncoding
from src.ml.models import SITSFormerClassifier


class SITSFormerModule(L.LightningModule):
    def __init__(
        self,
        n_bands: int = 10,
        d_model: int = 256,
        n_classes: int = 19,
        band_emb_chanels: List[int] = [32, 64],
        band_emb_kernel_size: List[int] = [5, 1, 10, 1],
        att_hidden_size: int = 256,
        n_att_head: int = 8,
        n_att_layers: int = 3,
        dropout_p: float = 0.1,
        lr: float = 2e-4,
        wd: float = 1e-4,
    ):
        # hyper parameters
        self.lr = lr
        self.wd = wd

        # layers and model
        band_emb_kernel_size[2] = n_bands - 4

        position_encoder = PositionalEncoding(d_model=d_model)
        bands_encoder = CNN3DEncoding(
            channel_size=band_emb_chanels.append(d_model),
            kernel_size=band_emb_kernel_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_att_head,
            dim_feedforward=att_hidden_size * 4,
            dropout=dropout_p,
        )
        encoder_norm = nn.LayerNorm(att_hidden_size)

        transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_att_layers,
            norm=encoder_norm,
        )

        clf_head = nn.Linear(att_hidden_size, n_classes)

        self.classifier = SITSFormerClassifier(
            position_encoder,
            bands_encoder,
            transformer_encoder,
            clf_head,
        )

        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        ts, days, mask, y = [batch[k] for k in ["ts", "days", "mask", "class"]]
        y_hat = self.classifier(ts, days, mask)
        loss = self.loss(y, y_hat)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ts, days, mask, y = [batch[k] for k in ["ts", "days", "mask", "class"]]
        y_hat = self.classifier(ts, days, mask)
        loss = self.loss(y, y_hat)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return Adam(self.classifier.parameters(), lr=self.lr, weight_decay=self.wd)
