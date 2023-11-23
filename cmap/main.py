import logging

from cli import SupervisedCLI
from data.modules import SITSDataModule
from ml.modules import SITSFormerModule

# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)


def cli_main():
    SupervisedCLI(model_class=SITSFormerModule, datamodule_class=SITSDataModule)


if __name__ == "__main__":
    cli_main()
