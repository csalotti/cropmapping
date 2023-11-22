from glob import glob
import logging
from os.path import join

import pandas as pd
import pytorch_lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from utils.constants import LABEL_COL, POINT_ID_COL, SEASON_COL

from data.dataset import ChunkDataset
from utils.chunk import chunks_indexing

logger = logging.getLogger("lightning.pytorch.core")


class SITSDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        prepare: bool = False,
        num_workers: int = 3,
        records_frac: float = 1.0,
    ):
        L.LightningDataModule.__init__(self)

        self.train_root = join(data_root, "train")
        self.val_root = join(data_root, "eval")
        self.batch_size = batch_size
        self.prepare = prepare
        self.num_workers = num_workers
        self.records_frac = records_frac

    def prepare_data(self) -> None:
        if self.prepare:
            chunks_indexing(join(self.train_root, "features"), write_csv=True)
            chunks_indexing(join(self.val_root, "features"), write_csv=True)

    def __retrieve(self, root: str):
        features_root = join(root, "features")
        labels_root = join(root, "labels")

        indexes = pd.read_json(join(features_root, "indexes.json"))
        labels = pd.concat(
            [pd.read_csv(f, index_col=0) for f in glob(join(labels_root, "*.csv"))]
        )

        if self.records_frac < 1.0:
            labels = labels.groupby([LABEL_COL, SEASON_COL], group_keys=False).apply(
                lambda x: x.sample(frac=self.records_frac)
            )
            indexes = indexes[indexes[POINT_ID_COL].isin(labels[POINT_ID_COL])]

        return indexes, labels

    def setup(self, stage: str):
        # Train
        train_features_root = join(self.train_root, "features")
        train_indexes, train_labels = self.__retrieve(self.train_root)
        self.train_dataset = ChunkDataset(
            features_root=train_features_root,
            labels=train_labels,
            indexes=train_indexes,
        )

        # Val
        val_features_root = join(self.val_root, "features")
        val_indexes, val_labels = self.__retrieve(self.val_root)
        self.val_dataset = ChunkDataset(
            features_root=val_features_root,
            labels=val_labels,
            indexes=val_indexes,
        )

    def train_dataloader(self):
        ds_shuffled = ShufflerIterDataPipe(
            self.train_dataset,
            buffer_size=self.batch_size * 10,
        )
        return DataLoader(
            ds_shuffled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
        )
