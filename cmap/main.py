import logging
from multiprocessing import set_start_method

import pytorch_lightning as L
from pytorch_lightning.cli import LightningCLI


# configure logging at the root level of Lightning
logging.getLogger("cmap").setLevel(logging.DEBUG)


def cli_main():
    LightningCLI(
        L.LightningModule,
        L.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    set_start_method("spawn")
    cli_main()
