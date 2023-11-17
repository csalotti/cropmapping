import logging
from pprint import pprint
from data import SITSDataModule

DATA_ROOT = "data"
logging.basicConfig(level=logging.INFO)


logger = logging.getLogger("main")


def main():
    dmodule = SITSDataModule(data_root=DATA_ROOT)
    dmodule.prepare_data()
    dmodule.setup("train")
    train_loader = dmodule.train_dataloader()
    dmodule.setup("eval")
    val_loader = dmodule.val_dataloader()

    pprint(next(iter(train_loader)))
    pprint(next(iter(val_loader)))


if __name__ == "__main__":
    main()
