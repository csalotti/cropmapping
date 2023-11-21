import logging

from pytorch_lightning.cli import LightningCLI

from data.modules import SITSDataModule
from ml.modules import SITSFormerModule

# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)

def cli_main():
    LightningCLI(model_class=SITSFormerModule, datamodule_class=SITSDataModule)


if __name__ == "__main__":
    cli_main()
