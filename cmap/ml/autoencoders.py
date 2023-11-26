import pytorch_lightning as L
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR


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

    def training_step(self, batch, batch_idx):
        ts, days, target, mask, loss_mask = [
            batch[k] for k in ["ts", "days", "target", "mask", "loss_mask"]
        ]

        ts_encoded = self.encoder(ts, days, mask)
        ts_hat = self.decoder(ts_encoded)

        loss = self.criterion(ts_hat, target)
        loss = (loss.mean(dim=-1) * loss_mask.float()).sum() / loss_mask.sum()

        self.log_dict(
            {
                "Losses/train": loss,
            },
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        ts, days, target, mask, loss_mask = [
            batch[k] for k in ["ts", "days", "target", "mask", "loss_mask"]
        ]

        ts_encoded = self.encoder(ts, days, mask)
        ts_hat = self.decoder(ts_encoded)

        loss = self.criterion(ts_hat, target)
        loss = (loss.mean(dim=-1) * loss_mask.float()).sum() / loss_mask.sum()

        self.log_dict(
            {
                "Losses/val": loss,
            },
            on_step=True,
            on_epoch=False,
        )

        return loss

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
