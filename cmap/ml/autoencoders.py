from random import random
import pytorch_lightning as L
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR

from cmap.utils.ndvi import plot_ndvi
from cmap.utils.attention import plot_attention, patch_attention


class AutoEncoder(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        criterion: nn.Module,
        min_lr: float = 1e-6,
        max_lr: float = 1e-4,
        gamma: float = 0.99,
        warmup_epochs: int = 10,
        wd: float = 1e-4,
        ndvi_sample: int = 10,
    ):
        super().__init__()

        # Optimizationin
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.wd = wd

        # Metrics
        self.criterion = criterion

        # Layers
        self.encoder = encoder
        self.decoder = decoder

        # data
        self.ndvi_sample = ndvi_sample
        self.val_data = []
        self.plot_indexes = []

        self.save_hyperparameters(ignore=["encoder", "decoder", "criterion"])

    def training_step(self, batch, batch_idx):
        ts, days, target, mask, loss_mask, _ = [
            batch[k] for k in ["ts", "days", "target", "mask", "loss_mask", "season"]
        ]

        ts_encoded = self.encoder(ts, days, mask)
        ts_hat = self.decoder(ts_encoded)

        loss = self.criterion(ts_hat, target)
        loss = (loss * loss_mask.unsqueeze(-1).float()).sum() / loss_mask.sum()

        self.log_dict(
            {
                "Losses/train": loss,
            },
            on_step=False,
            on_epoch=True,
        )
        return loss

    def _get_attention_maps(self, ts, days, mask):
        for l in self.encoder.transformer_encoder.layers:
            patch_attention(l.self_attn)

        return self.encoder.get_attention_maps(ts, days, mask)

    def validation_step(self, batch, batch_idx):
        ts, days, target, mask, loss_mask, seasons = [
            batch[k] for k in ["ts", "days", "target", "mask", "loss_mask", "season"]
        ]

        ts_encoded = self.encoder(ts, days, mask)
        ts_hat = self.decoder(ts_encoded)

        loss = self.criterion(ts_hat, target)
        loss = (loss * loss_mask.unsqueeze(-1).float()).sum() / loss_mask.sum()

        self.log_dict(
            {
                "Losses/val": loss,
            },
            on_step=False,
            on_epoch=True,
        )

        if len(self.val_data) == 0:
            batch_size = ts.shape[0]

            if len(self.plot_indexes) == 0:
                self.plot_indexes = np.random.choice(
                    range(batch_size), min(self.ndvi_sample, batch_size)
                )

            attn_maps = self._get_attention_maps(ts, days, mask)
            self.val_data.append(
                {
                    "target": target.cpu().numpy()[self.plot_indexes, :, :],
                    "ts_hat": ts_hat.cpu().numpy()[self.plot_indexes, :, :],
                    "days": days.cpu().numpy()[self.plot_indexes, :],
                    "mask": mask.cpu().numpy()[self.plot_indexes, :],
                    "attn_maps": attn_maps.cpu().numpy()[self.plot_indexes, :],
                    "loss_mask": loss_mask.cpu().numpy()[self.plot_indexes, :],
                    "season": seasons.cpu().numpy()[self.plot_indexes, :],
                }
            )

        return loss

    def on_validation_end(self) -> None:
        if len(self.val_data) > 0:
            batch = self.val_data.pop()
            batch_size = batch["target"].shape[0]
            for i in range(batch_size):
                ndvi_fig = plot_ndvi(
                    days=batch["days"][i],
                    days_mask=batch["mask"][i],
                    removed_days_mask=batch["loss_mask"][i],
                    pred=batch["ts_hat"][i],
                    gt=batch["target"][i],
                    start_year=int(batch["season"][i][0]) - 1,
                )

                att_fig = plot_attention(
                    attention_map=batch["attn_maps"][i],
                    days=batch["days"][i],
                    mask=batch["mask"][i],
                )

                self.logger.experiment.add_figure(
                    f"NDVI/sample_{i}",
                    ndvi_fig,
                    self.current_epoch,
                )

                self.logger.experiment.add_figure(
                    f"Attention Maps/sample_{i}",
                    att_fig,
                    self.current_epoch,
                )

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
