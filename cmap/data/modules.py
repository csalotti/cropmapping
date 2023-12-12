import logging
from typing import List, Dict

import pandas as pd
import pytorch_lightning as L
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

import os
from cmap.data.transforms import labels_sample
from cmap.data.dataset import SITSDataset
from cmap.utils.constants import LABEL_COL


logger = logging.getLogger("cmap.data.module")
# logger.addHandler(logging.FileHandler("datamodule.log"))


class SITSDataModule(L.LightningDataModule):
    """ """

    def __init__(
        self,
        root: str,
        classes: List[str],
        train_seasons: List[int] = [2017, 2018, 2019, 2020],
        val_seasons: List[int] = [2021],
        extra_features: Dict[str, Dict[str, str]] = {},
        rpg_mapping: str = "",
        fraction: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 3,
        chunk_size: int = 10,
    ):
        L.LightningDataModule.__init__(self)

        # Data
        self.root = root
        self.classes = classes
        self.train_seasons = train_seasons
        self.val_seasons = val_seasons
        self.extra_features = extra_features
        self.rpg_mapping = rpg_mapping

        # Hyperparams
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size

        # Subsampling
        self.fraction = fraction

    def get_dataset(self, stage: str, seasons: List[int]):
        # Gather
        features_file = os.path.join(self.root, stage, "features.pq")
        labels = pd.read_parquet(os.path.join(self.root, stage, "labels.pq"))
        extra_features_files = {
            fn: self.extra_features[fn][stage] for fn in self.extra_features.keys()
        }

        # Filter
        if self.rpg_mapping is not None:
            labels[LABEL_COL] = labels[LABEL_COL].map(
                lambda x: self.rpg_mapping.get(x, "other")
            )
        labels = labels_sample(
            labels, seasons=seasons, classes=self.classes, fraction=self.fraction
        )

        return SITSDataset(
            features_file=features_file,
            labels=labels,
            classes=self.classes,
            augment=stage == "train",
            extra_features_files=extra_features_files,
        )

    def setup(self, stage: str):
        if stage == "fit":
            if self.rpg_mapping:
                self.rpg_mapping = yaml.safe_load(open(self.rpg_mapping, "r"))
                self.rpg_mapping = {
                    ri: ci for ci, rpgs in self.rpg_mapping.items() for ri in rpgs
                }

            self.train_dataset = self.get_dataset("train", self.train_seasons)
            self.val_dataset = self.get_dataset("val", self.val_seasons)

    def train_dataloader(self):
        ds_shuffled = ShufflerIterDataPipe(
            self.train_dataset,
            buffer_size=100,
        )
        return DataLoader(
            ds_shuffled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
