import logging
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn as nn
from safetensors.torch import load_model
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score

from cmap.utils.attention import merge, patch_attention, plot_attention, resample
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
    """Classifier Lightning Module
    Core system for model training and validation. It encapsulate training and
    validation loop and provide logs and plots for attention, distribution and performances.
    Dependency injection is used for layers to offer a greater fllexibility and smaller
    savings.
    More details on extensibility on pytorch_lightning documentation.

    Attributes:
        encoder (nn.Module) : Sequence Encoder
        decoder (nn.Module) : Sequence Decoder, mainly attention head
        classes (List[str]) : Expected classes
        n_classes (int) : Number of classes
        classes_weights (FloatTensor): Classes weights for CrossEntropyLoss
        min_lr/max_lr (float) : Learning rate limit for warm up
        gamma (float) : LR reduction factor
        warmup_epochs (int) : Number of epochs for warmup
        wd (float) : Adam optimizer weight decay
        criterion :(nn.CrossEntropyLoss) : Loss
        train_f1 (MulticlassF1Score) : Training Scoring entity
        train_conf_mat (MulticlassConfusionMatrix) : Training confusion matrix
        val_f1 (MulticlassF1Score) : Validation Scoring entity
        val_conf_mat (MulticlassConfusionMatrix) : Validation confusion matrix
        val_emb (List[Tensor]) : Last layer features embeddings
        batch_attn (List[numpy.ndarray]) : Transformer encoder attention maps
        train_labels (List[NDArray]) : Training Labels
        val_labels (List[NDArray]) : Validation Labels
        patched (bool) : Flag for Transformer attention patching

    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        classes_weights: Optional[Dict[str, float]] = None,
        encoder_weights_path: str = "",
        classes: List[str] = DEFAULT_CLASSES,
        min_lr: float = 1e-6,
        max_lr: float = 1e-4,
        gamma: float = 0.99,
        warmup_epochs: int = 10,
        wd: float = 1e-4,
    ):
        """Classifier initialization

        Args:
            encoder (nn.Module) : Sequence Encoder
            decoder (nn.Module) : Sequence Decoder, mainly attention head
            classes (List[str]) : Expected classes
            n_classes (int) : Number of classes
            classes_weights (Optional[Dict[str, float]]) : Classes weights for CrossEntropyLoss
            min_lr/max_lr (float) : Learning rate limit for warm up
            gamma (float) : LR reduction factor
            warmup_epochs (int) : Number of epochs for warmup
            wd (float) : Adam optimizer weight decay

        """
        super().__init__()

        # labels
        self.classes = classes
        self.n_classes = len(classes)
        self.classes_weights = (
            torch.FloatTensor([classes_weights[c] for c in classes])
            if (classes_weights is not None)
            else None
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

        # Flags
        self.patched = False

        # Layers
        self.encoder = encoder
        self.decoder = decoder

        if encoder_weights_path:
            load_model(self.encoder, encoder_weights_path)

        self.save_hyperparameters(ignore=["encoder", "decoder"])

    def training_step(self, batch: Dict[str, torch.Tensor], _):
        """Perform a single training step (single batch).
        Necessary data for plots and logs are stored during this phase

        Notes on shapes:
            N : Batch size
            S : Sequence size
            B : Number of bandsk
            E : Final band embeddings size

        Args :
            batch (Dict[str], Tensor) : Batch data with following content :
                ts [N, S, B] : Sequencee with bands features
                positions [N, S] : sequence positions
                mask [N, S] : sequencee mask (`1` indicates time step is valid)
                class [N ,1] : classes indexes

        Returns :
            Criterion Loss
        """
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
        self.train_f1.update(y_hat_cls, y)
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
        """Training epoch end computation.
        Retrieved data stored during training, plots confusion and distribution
        and log metrics
        """
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
        """Retrieve attention from transfomer encoder
        If not done, nn.TransformerEncoder is patched to return
        attention maps, wich isn't the case by default

        Notes on shapes:
            N : Batch size
            S : Sequence size
            B : Number of S2 bands
            L : Number of attention layers

        Args:
            ts [N, S, B] : Sequencee with bands features
            positions [N, S] : sequence positions
            mask [N, S] : sequencee mask (`1` indicates time step is valid)

        Returns
            Attnetion Maps per layers ([N, L, S, S])
        """
        if not self.patched:
            for l in self.encoder.transformer_encoder.layers:
                patch_attention(l.self_attn)
            self.patched = True

        return self.encoder.get_attention_maps(ts, positions, mask)

    def validation_step(self, batch, _):
        """Perform a single validation step (single batch).

        Notes on shapes:
            N : Batch size
            S : Sequence size
            B : Number of bandsk
            E : Final band embeddings size

        Args :
            batch (Dict[str], Tensor) : Batch data with following content :
                ts [N, S, B] : Sequencee with bands features
                positions [N, S] : sequence positions
                days [N, S] : sequence days
                mask [N, S] : sequencee mask (`1` indicates time step is valid)
                class [N ,1] : classes indexes

        """
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
        self.val_f1.update(y_hat_cls, y)

        self.log(
            "Losses/val",
            loss,
            on_step=False,
            on_epoch=True,
        )

        self.val_conf_mat.update(y_hat, y)

        # Save embeddings & Attention Maps
        if self.current_epoch % 10 == 1:
            self.val_emb.append(
                {
                    "embeddings": y_hat.cpu().numpy(),
                    "labels": [self.classes[yi] for yi in y.cpu().tolist()],
                }
            )
            attn_maps = self._get_attention_maps(ts, positions, mask)
            self.batch_attn.append(
                resample(
                    attention_map=attn_maps.cpu().numpy(),
                    days=days.cpu().numpy(),
                    masks=mask.cpu().numpy(),
                    targets=y.cpu().numpy(),
                )
            )

        # Save Labels
        if self.current_epoch == 0:
            self.val_labels.extend([self.classes[i] for i in y.cpu().tolist()])

    def on_validation_epoch_end(self):
        """Validiation epoch end computation.
        Retrieved data stored during training, plots attention, confusion and distribution
        and log metrics
        """
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
                        class_attn_maps_df,
                        "month",
                        ci,
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
        """Training optimizers config.
        A global Adam is is used with a LRScheduler with a warmup
        for the first warmup epochs.
        The warmup consists of a linear increase from min_lr to max_lr
        and the remaining is a ExponentialLR with a reduction factor of gamma

        """
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
