import logging
from typing import List, Dict, Optional

from safetensors.torch import load_model
import numpy as np
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score


from cmap.ml.losses import FocalLoss
from cmap.utils.attention import patch_attention, plot_attention, resample, merge
from cmap.utils.distributions import get_dist_plot

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
        class_weights: Optional[Dict[str, float]] = None,
        encoder_weights_path: str = "",
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
        self.classes_weights = torch.FloatTensor(
            [class_weights[c] for c in classes] if (class_weights is not None) else None
        )

        # Optimizationin
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.wd = wd

        # Metrics
        self.criterion = nn.CrossEntropyLoss(weight=self.classes_weights)
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
        self.batch_attn = []
        self.train_labels = []
        self.val_labels = []

        # Layers
        self.encoder = encoder
        self.decoder = decoder

        if encoder_weights_path:
            load_model(self.encoder, encoder_weights_path)

        self.save_hyperparameters(ignore=["encoder", "decoder"])

    def training_step(self, batch, batch_idx):
        ts, positions, mask, y = [
            batch[k] for k in ["ts", "positions", "mask", "class"]
        ]

        # Infer
        ts_encoded = self.encoder(ts=ts, positions=positions, mask=mask)
        y_hat = self.decoder(x=ts_encoded, mask=mask)

        # Reshape
        y = y.squeeze()
        _, y_hat_cls = torch.max(y_hat, dim=1)

        # Metrics
        loss = self.criterion(y_hat, y)
        f1_score = self.train_f1.update(y_hat_cls, y)
        self.train_conf_mat.update(y_hat, y)

        # Logging
        self.log(
            "Losses/train",
            loss,
            on_step=False,
            on_epoch=True,
        )

        # Data
        # Save Labels
        if self.current_epoch == 0:
            self.train_labels.extend([self.classes[i] for i in y.cpu().tolist()])

        return loss

    def on_train_epoch_end(self):
        cmat_fig = sns.heatmap(
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
            cmat_fig,
            self.current_epoch,
        )

        self.logger.experiment.add_scalars(
            "F1",
            {"train": self.train_f1.compute()},
            global_step=self.current_epoch,
        )

        # Label distribution
        if (self.current_epoch == 0) and (len(self.train_labels) > 0):
            labels_dist_fig = get_dist_plot(self.train_labels)
            self.logger.experiment.add_figure(
                "Labels Distribution/Training",
                labels_dist_fig,
                global_step=self.current_epoch,
            )

        self.train_f1.reset()
        self.train_conf_mat.reset()
        self.train_labels.clear()

    def _get_attention_maps(self, ts, positions, mask):
        for l in self.encoder.transformer_encoder.layers:
            patch_attention(l.self_attn)

        return self.encoder.get_attention_maps(ts, positions, mask)

    def validation_step(self, batch, batch_idx):
        ts, positions, days, mask, y = [
            batch[k] for k in ["ts", "positions", "days", "mask", "class"]
        ]

        # Infer
        ts_encoded = self.encoder(ts=ts, positions=positions, mask=mask)
        y_hat = self.decoder(x=ts_encoded, mask=mask)

        # Reshape
        y = y.squeeze()
        _, y_hat_cls = torch.max(y_hat, dim=1)

        # Metrics
        loss = self.criterion(y_hat, y)
        f1_score = self.val_f1.update(y_hat_cls, y)

        self.log(
            "Losses/val",
            loss,
            on_step=False,
            on_epoch=True,
        )

        self.val_conf_mat.update(y_hat, y)

        # Save embeddings
        if self.current_epoch % 10 == 1:
            self.val_emb.append(
                {
                    "embeddings": y_hat.cpu().numpy(),
                    "labels": [self.classes[yi] for yi in y.cpu().tolist()],
                }
            )

        # Save data for attention maps
        if (self.current_epoch > 0) or (batch_idx == 0):
            attn_maps = self._get_attention_maps(ts, positions, mask)
            self.batch_attn.append(
                resample(
                    attention_map=attn_maps.cpu().numpy(),
                    days=days.cpu().numpy(),
                    masks=mask.cpu().numpy(),
                    targets=y.cpu().numpy(),
                )
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
            "F1",
            {"val": self.val_f1.compute()},
            global_step=self.current_epoch,
        )

        if self.current_epoch % 10 == 1:
            emb = np.concatenate([d["embeddings"] for d in self.val_emb], axis=0)
            metadata = [di for d in self.val_emb for di in d["labels"]]
            self.logger.experiment.add_embedding(
                emb,
                metadata=metadata,
                global_step=self.current_epoch,
            )

        if len(self.batch_attn) > 0:
            batch_attn_maps_df = merge(self.batch_attn)

            for i, ci in enumerate(self.classes):
                class_attn_maps_df = batch_attn_maps_df.query(f"target == {i}")
                if len(class_attn_maps_df) > 0:
                    attn_fig = plot_attention(
                         class_attn_maps_df, "month", ci,
                    )

                    self.logger.experiment.add_figure(
                        f"Attention Maps/{ci}",
                        attn_fig,
                        self.current_epoch,
                    )

        # Label distribution
        if (self.current_epoch == 0) and (len(self.val_labels) > 0):
            labels_dist_fig = get_dist_plot(self.val_labels)
            self.logger.experiment.add_figure(
                "Labels Distribution/Validation",
                labels_dist_fig,
                global_step=self.current_epoch,
            )

        self.val_labels.clear()
        self.batch_attn.clear()
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
