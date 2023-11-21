from pytorch_lightning.cli import LightningCLI

from data.modules import SITSDataModule
from ml.modules import SITSFormerModule


def cli_main():
    LightningCLI(model_class=SITSFormerModule, datamodule_class=SITSDataModule)


if __name__ == "__main__":
    cli_main()
