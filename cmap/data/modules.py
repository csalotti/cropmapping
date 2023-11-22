from glob import glob
import logging
from os.path import join

import pandas as pd
import pytorch_lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
import yaml

from utils.constants import LABEL_COL, POINT_ID_COL, SEASON_COL
from data.dataset import ChunkDataset, CLASSES
from utils.chunk import chunks_indexing

logger = logging.getLogger("lightning.pytorch.core")


class SITSDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        classes_config: str = "configs/rpg_codes.yml",
        batch_size: int = 32,
        prepare: bool = False,
        num_workers: int = 3,
        records_frac: float = 1.0,
    ):
        L.LightningDataModule.__init__(self)

        self.train_root = join(data_root, "train")
        self.val_root = join(data_root, "eval")
        self.classes_config = classes_config
        self.batch_size = batch_size
        self.prepare = prepare
        self.num_workers = num_workers
        self.records_frac = records_frac

    def prepare_data(self) -> None:
        with open(self.classes_config, "r") as f:
            class_to_label = yaml.safe_load(f)
            self.label_to_class = {v: k for k, vs in class_to_label.items() for v in vs}

        if self.prepare:
            chunks_indexing(join(self.train_root, "features"), write_csv=True)
            chunks_indexing(join(self.val_root, "features"), write_csv=True)

    def __retrieve(self, root: str):
        features_root = join(root, "features")
        labels_root = join(root, "labels")

        indexes = pd.read_json(join(features_root, "indexes.json"))

        # Select subset of available classes
        labels = pd.concat(
            [pd.read_csv(f, index_col=0) for f in glob(join(labels_root, "*.csv"))]
        )
        labels[LABEL_COL] = labels[LABEL_COL].map(
            lambda x: self.label_to_class.get(x, "other")
        )
        labels = labels.query(f"{LABEL_COL} in {CLASSES}")

        # Subsample dataset respecting distribution of classes
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
            label_to_class=self.label_to_class,
        )

        # Val
        val_features_root = join(self.val_root, "features")
        val_indexes, val_labels = self.__retrieve(self.val_root)
        self.val_dataset = ChunkDataset(
            features_root=val_features_root,
            labels=val_labels,
            indexes=val_indexes,
            label_to_class=self.label_to_class,
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
