import logging
import os
from typing import List, Dict

import pandas as pd
import pytorch_lightning as L
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

from cmap.data.dataset import SITSDataset
from cmap.data.transforms import labels_sample
from cmap.utils.constants import LABEL_COL

logger = logging.getLogger("cmap.data.module")
# logger.addHandler(logging.FileHandler("datamodule.log"))


class SITSDataModule(L.LightningDataModule):
    """Satelite Image Time Seriess Data module

    Handles data preparation, preprocessing and create corresponding
    Dataset and DataLoader.
    All preprocessing is detailed to prepare and feed data to the model
    It supposed that the data is in a folder with the following structure

        root
         ├── train
         │    ├── features.pq
         │    ├── labels.pq
         │    └── <other_features>.pq
         └── val
              ├── features.pq
              ├── labels.pq
              └── <other_features>.pq

    Attributes:
        root (str): Data root
        classes (List[str) : Expected classes
        train_seasons (List[int]) : Training seasons
        val_seasons (List[int] : Validation seasons
        extra_features (Dict[str, Dict[str, str | List[str]]]) : Maps of
            additionnal files with their name and path for train and validation datasets.
            The dictionnary must have the following structure:
            {'temperatures' : {'file' : <fname> , 'features_cols' : [<f1>, <f2>]}}
        rpg_mapping (Dict[str, str])=  RPG CODE to label mapping
        fraction (float) : Dataset sampling fraction
        batch_size (int) : Batch size
        num_workers (int) : Number of parallell Dataloader workers
    """

    def __init__(
        self,
        root: str,
        classes: List[str],
        train_seasons: List[int] = [2017, 2018, 2019, 2020],
        val_seasons: List[int] = [2021],
        extra_features: Dict[str, Dict[str, str | List[str]]] = {},
        rpg_mapping_path: str = "",
        fraction: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 3,
    ):
        """Initialize Lightning Data Module for SITS
        Args:
            root (str): Data root
            classes (List[str) : Expected classes
            train_seasons (List[int]) : Training seasons
            val_seasons (List[int] : Validation seasons
            extra_features (Dict[str, Dict[str, str | List[str]]]) : Maps of
                additionnal files with their name and path for train and validation datasets.
                The dictionnary must have the following structure:
                {'temperatures' : {'file' : <fname> , 'features_cols' : [<f1>, <f2>]}}
            rpg_mapping_path (str) = path to mapping yaml
            fraction (float) : Dataset sampling fraction
            batch_size (int) : Batch size
            num_workers (int) : Number of parallell Dataloader workers
        """
        L.LightningDataModule.__init__(self)

        # Data
        self.root = root
        self.classes = classes
        self.train_seasons = train_seasons
        self.val_seasons = val_seasons
        self.extra_features = extra_features
        self.rpg_mapping = yaml.safe_load(open(rpg_mapping_path, "r"))

        # Hyperparams
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Subsampling
        self.fraction = fraction

    def get_dataset(self, stage: str, seasons: List[int]):
        """Retrieve features and label with respect to stage
        to createe the dataset. RPG codes are mapped to ttheir
        label if provided, then filtered and sampled

        Args :
            stage (str) : Stage name (train or val)
            seasons (List[int]) : Focused seasons

        Returns:
            SITSDataset for the respective stage and seeasons
                with the provided features
        """
        # Gather
        features_file = os.path.join(self.root, stage, "features.pq")
        labels = pd.read_parquet(os.path.join(self.root, stage, "labels.pq"))
        extra_features_files = {
            fn: {
                "path": os.path.join(self.root, stage, f_conf["file"]),
                "features": f_conf["features_cols"],
            }
            for fn, f_conf in self.extra_features.items()
        }

        # Filter
        if self.rpg_mapping is not None:
            labels[LABEL_COL] = labels[LABEL_COL].map(self.rpg_mapping).fillna("other")

        labels = labels_sample(
            labels, seasons=seasons, classes=self.classes, fraction=self.fraction
        )

        return SITSDataset(
            features_file=features_file,
            labels=labels,
            classes=self.classes,
            augment=False,
            extra_features_files=extra_features_files,
        )

    def setup(self, stage: str):
        """Data preparation and preprocessing. RPG mmapping is
        generateed if provideed and datasets are created.

        Args:
            stage (str) :Pytorch Lightning stage name (fit, predict, test)

        """
        if stage == "fit":
            if self.rpg_mapping:
                self.rpg_mapping = {
                    ri: ci for ci, rpgs in self.rpg_mapping.items() for ri in rpgs
                }

            self.train_dataset = self.get_dataset("train", self.train_seasons)
            self.val_dataset = self.get_dataset("val", self.val_seasons)

    def train_dataloader(self):
        """Train Data Loader creation"""
        ds_shuffled = ShufflerIterDataPipe(
            self.train_dataset,
            buffer_size=100,
        )
        return DataLoader(
            ds_shuffled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        """Validation Data Loader creation"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
