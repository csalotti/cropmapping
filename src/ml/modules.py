from typing import Any, List, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR, SequentialLR
from torchmetrics.classification import F1Score

from ml.embeddings.bands import PatchBandsEncoding
from ml.embeddings.position import PositionalEncoding
from ml.losses import FocalLoss
from ml.models import MulticlassClassification, SITSFormerClassifier


class SITSFormerModule(L.LightningModule):
    def __init__(
        self,
        n_bands: int = 9,
        max_n_days: int = 397,
        d_model: int = 256,
        n_classes: int = 19,
        band_emb_chanels: List[int] = [32, 64],
        band_emb_kernel_size: List[int] = [5, 1, 5, 1],
        att_hidden_size: int = 256,
        n_att_head: int = 8,
        n_att_layers: int = 3,
        dropout_p: float = 0.1,
        lr: float = 1e4,
        gamma: float = 0.99,
        warmup_epochs: int = 10,
        wd: float = 1e-4,
        decay_max_epoch: int = 50,
    ):
        L.LightningModule.__init__(self)

        # hyper parameters
        self.lr = lr
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.wd = wd
        self.decay_max_epoch = decay_max_epoch

        # layers and model
        band_emb_kernel_size[2] = n_bands - 4

        position_encoder = PositionalEncoding(d_model=d_model, max_len=max_n_days)
        bands_encoder = PatchBandsEncoding(
            channel_size=band_emb_chanels + [d_model],
            kernel_size=band_emb_kernel_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_att_head,
            dim_feedforward=att_hidden_size * 4,
            dropout=dropout_p,
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(att_hidden_size)

        transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_att_layers,
            norm=encoder_norm,
        )

        clf_head = MulticlassClassification(att_hidden_size, n_classes)

        self.classifier = SITSFormerClassifier(
            position_encoder,
            bands_encoder,
            transformer_encoder,
            clf_head,
        )

        self.criterion = FocalLoss(gamma=1)
        self.scorer = F1Score(task="multiclass", num_classes=n_classes)

    def training_step(self, batch, batch_idx):
        ts, days, mask, y = [batch[k] for k in ["ts", "days", "mask", "class"]]
        y_hat = self.classifier(ts, days, mask)
        y = y.squeeze()
        _, y_hat_cls = torch.max(y_hat, dim=1)
        loss = self.criterion(y_hat, y)
        f1_score = self.scorer(y_hat_cls, y)
        self.log("train_loss", loss)
        self.log("train_f1_score", f1_score)
        return loss

    def validation_step(self, batch, batch_idx):
        ts, days, mask, y = [batch[k] for k in ["ts", "days", "mask", "class"]]
        y_hat = self.classifier(ts, days, mask)
        y = y.squeeze()
        _, y_hat_cls = torch.max(y_hat, dim=1)
        loss = self.criterion(y_hat, y)
        f1_score = self.scorer(y_hat_cls, y)
        self.log("val_loss", loss)
        self.log("val_f1_score", f1_score)

    def configure_optimizers(self):
        optimizer = Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
        )
        warmup_lr_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=self.warmup_epochs,
        )
        decreasing_scheduler = ExponentialLR(
            optimizer=optimizer,
            gamma=self.gamma,
        )
        lr_scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_lr_scheduler, decreasing_scheduler],
            milestones=[self.warmup_epochs],
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}