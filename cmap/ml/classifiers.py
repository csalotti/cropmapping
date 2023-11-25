import logging
from typing import List

import numpy as np
import seaborn as sns
import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from ml.losses import FocalLoss

logger = logging.getLogger("cmap.ml.modules")
# logger.addHandler(logging.FileHandler("cmap.ml.modules.log"))

DEFAULT_CLASSES = [
    "other",
    "ble_dur",
    "ble_tendre",
    "orge",
    "colza",
    "mais",
    "tournesol",
]


class Classifier(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        classes: List[str] = DEFAULT_CLASSES,
        min_lr: float = 1e-6,
        max_lr: float = 1e-4,
        gamma: float = 0.99,
        warmup_epochs: int = 10,
        wd: float = 1e-4,
    ):
        super().__init__()

        # labels
        self.classes = classes
        self.n_classes = len(classes)

        # Optimizationin
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.wd = wd

        # Metrics
        self.criterion = FocalLoss(gamma=1)
        self.train_f1 = MulticlassF1Score(num_classes=len(classes))
        self.val_f1 = MulticlassF1Score(num_classes=len(classes))
        self.train_conf_mat = MulticlassConfusionMatrix(
            num_classes=len(classes),
            normalize="true",
        )
        self.val_conf_mat = MulticlassConfusionMatrix(
            num_classes=len(classes),
            normalize="true",
        )

        # data
        self.val_emb = []

        # Layers
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        ts, days, mask, y = [batch[k] for k in ["ts", "days", "mask", "class"]]

        # Infer
        ts_encoded = self.encoder(ts, days, mask)
        y_hat = self.decoder(ts_encoded, mask)

        # Reshape
        y = y.squeeze()
        _, y_hat_cls = torch.max(y_hat, dim=1)

        # Metrics
        loss = self.criterion(y_hat, y)
        f1_score = self.train_f1(y_hat_cls, y)
        self.train_conf_mat.update(y_hat, y)

        logger.debug(batch)
        self.log_dict(
            {
                "Losses/train": loss,
                "F1/train": f1_score,
            },
            on_step=True,
            on_epoch=False,
        )

        return loss

    def on_train_epoch_end(self):
        fig_ = sns.heatmap(
            self.train_conf_mat.compute().cpu().numpy(),
            cmap="magma",
            annot=True,
            fmt=".2f",
            cbar=False,
            xticklabels=self.classes,
            yticklabels=self.classes,
        ).get_figure()

        self.logger.experiment.add_figure(
            "Confusion matrix/Training",
            fig_,
            self.current_epoch,
        )

        self.logger.experiment.add_scalars(
            "F1/epoch",
            {"train": self.train_f1.compute()},
            global_step=self.current_epoch,
        )
        self.train_f1.reset()
        self.train_conf_mat.reset()

    def validation_step(self, batch, batch_idx):
        ts, days, mask, y = [batch[k] for k in ["ts", "days", "mask", "class"]]

        # Infer
        ts_encoded = self.encoder(ts, days, mask)
        y_hat = self.decoder(ts_encoded, mask)

        # Reshape
        y = y.squeeze()
        _, y_hat_cls = torch.max(y_hat, dim=1)

        # Metrics
        loss = self.criterion(y_hat, y)
        f1_score = self.val_f1(y_hat_cls, y)

        self.log_dict(
            {
                "Losses/val": loss,
                "F1/val": f1_score,
            },
            on_step=True,
            on_epoch=False,
        )

        self.val_conf_mat.update(y_hat, y)

        if self.current_epoch % 10 == 0:
            self.val_emb.append(
                {
                    "embeddings": y_hat.cpu().numpy(),
                    "labels": [self.classes[yi] for yi in y.cpu().tolist()],
                }
            )

    def on_validation_epoch_end(self):
        fig_ = sns.heatmap(
            self.val_conf_mat.compute().cpu().numpy(),
            cmap="magma",
            annot=True,
            fmt=".2f",
            cbar=False,
            xticklabels=self.classes,
            yticklabels=self.classes,
        ).get_figure()

        self.logger.experiment.add_figure(
            "Confusion matrix/Validation",
            fig_,
            self.current_epoch,
        )
        self.logger.experiment.add_scalars(
            "F1/epoch",
            {"val": self.val_f1.compute()},
            global_step=self.current_epoch,
        )

        if self.current_epoch % 10 == 0:
            emb = np.concatenate([d["embeddings"] for d in self.val_emb], axis=0)
            metadata = [di for d in self.val_emb for di in d["labels"]]
            self.logger.experiment.add_embedding(
                emb,
                metadata=metadata,
                global_step=self.current_epoch,
            )

        self.val_f1.reset()
        self.val_conf_mat.reset()
        self.val_emb.clear()

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.max_lr,
            weight_decay=self.wd,
        )
        warmup_lr_scheduler = LinearLR(
            optimizer,
            start_factor=self.min_lr / self.max_lr,
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