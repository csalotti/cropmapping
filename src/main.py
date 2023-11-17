import logging
from pprint import pprint
from time import time
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

    iter_train = iter(train_loader)
    iter_val = iter(val_loader)

    start = time()
    pprint({k: v.shape for k, v in next(iter_train).items()})
    print(f"Train Elapsed time : {time() - start}")
    start = time()
    pprint({k: v.shape for k, v in next(iter_val).items()})
    print(f"Val Elapsed time : {time() - start}")


if __name__ == "__main__":
    main()
