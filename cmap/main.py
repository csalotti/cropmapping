import logging


import pytorch_lightning as L
from pytorch_lightning.cli import LightningCLI

from data.modules import SITSDataModule

# configure logging at the root level of Lightning
logging.getLogger("cmap").setLevel(logging.DEBUG)


def cli_main():
    LightningCLI(
        L.LightningModule,
        SITSDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    cli_main()
