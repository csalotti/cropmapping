import logging
from os.path import join

import pandas as pd
import pytorch_lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

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
    ):
        L.LightningDataModule.__init__(self)

        self.train_root = join(data_root, "train")
        self.eval_root = join(data_root, "eval")
        self.batch_size = batch_size
        self.prepare = prepare
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        if self.prepare:
            chunks_indexing(join(self.train_root, "features"), write_csv=True)
            chunks_indexing(join(self.eval_root, "features"), write_csv=True)

    def setup(self, stage: str):
        features_root = join(self.train_root, "features")
        labels_root = join(self.train_root, "labels")
        indexes = pd.read_json(join(features_root, "indexes.json"))
        self.train_dataset = ChunkDataset(
            features_root=features_root,
            labels_root=labels_root,
            indexes=indexes,
        )

        features_root = join(self.eval_root, "features")
        labels_root = join(self.eval_root, "labels")
        indexes = pd.read_json(join(features_root, "indexes.json"))
        self.eval_dataset = ChunkDataset(
            features_root=features_root,
            labels_root=labels_root,
            indexes=indexes,
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
