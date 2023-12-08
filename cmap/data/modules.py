from collections import defaultdict
import json
import logging
from glob import glob
from os.path import join, basename
import shutil
from typing import List
from functools import partial

import pandas as pd
import pytorch_lightning as L
import multiprocessing
from sqlalchemy import create_engine
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

from cmap.data.chunk import ChunkLabeledDataset, ChunkMaskedDataset
import os
from glob import glob
from cmap.data.sql import SQLDataset
from cmap.data.transforms import labels_sample
from cmap.utils.chunk import preprocessing
from cmap.data.dataset import SITSDataset
from cmap.utils.constants import (
    LABEL_COL,
    POINT_ID_COL,
    SEASON_COL,
    CHUNK_ID_COL,
    DATE_COL,
)

logger = logging.getLogger("cmap.data.module")
# logger.addHandler(logging.FileHandler("datamodule.log"))


class CSVDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_features_root: str,
        val_features_root: str,
        batch_size: int = 32,
        prepare: bool = False,
        num_workers: int = 3,
        raw_data_root: str = os.environ.get("RAW_DATA_ROOT", ""),
    ):
        L.LightningDataModule.__init__(self)

        self.features_roots = {
            "train": train_features_root,
            "eval": val_features_root,
        }
        self.batch_size = batch_size
        self.prepare = prepare
        self.raw_data_root = raw_data_root
        self.num_workers = num_workers

    def _prepare_chunk(self, chunk_file, stage):
        features_df = pd.read_csv(chunk_file, index_col=0, parse_dates=[DATE_COL])
        chunk_id = features_df[CHUNK_ID_COL].iloc[0]

        indexes, features_df = preprocessing(features_df)

        features_df.to_csv(
            join(self.features_roots[stage], f"chunk_{chunk_id}.csv"),
        )
        return indexes

    def prepare_data(self) -> None:
        if self.prepare:
            for stage in ["train", "eval"]:
                os.makedirs(self.features_roots[stage], exist_ok=True)

                with multiprocessing.Pool(int(os.environ.get("N_PROCESSES", 10))) as p:
                    chunk_files = glob(
                        join(self.raw_data_root, stage, "features", "*.csv")
                    )
                    prepare_fn = partial(self._prepare_chunk, stage=stage)
                    chunk_indexes = p.map(prepare_fn, chunk_files)

                indexes = []
                for ci in chunk_indexes:
                    indexes.extend(ci)

                with open(join(self.features_roots[stage], "indexes.json"), "w") as f:
                    json.dump(indexes, f)

    def setup(self, stage: str) -> None:
        raise NotImplementedError("Datamodule subclasses must implement setup")

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


class LabelledDataModule(CSVDataModule):
    def __init__(
        self,
        data_root: str,
        classes: List[str],
        train_seasons: List[int] = [2017, 2018, 2019, 2020],
        val_seasons: List[int] = [2021],
        classes_config: str = "configs/rpg_codes.yml",
        use_temp: bool = False,
        batch_size: int = 32,
        prepare: bool = False,
        num_workers: int = 3,
        records_frac: float = 1,
        subsample: bool = False,
        raw_data_root: str = os.environ.get("RAW_DATA_ROOT", ""),
    ):
        super().__init__(
            train_features_root=join(data_root, "train", "features"),
            val_features_root=join(data_root, "eval", "features"),
            batch_size=batch_size,
            prepare=prepare,
            num_workers=num_workers,
            raw_data_root=raw_data_root,
        )

        self.labels_roots = {
            "train": join(data_root, "train", "labels"),
            "eval": join(data_root, "eval", "labels"),
        }
        self.seasons = {
            "train": train_seasons,
            "eval": val_seasons,
        }
        self.classes = classes
        self.classes_config = classes_config
        self.use_temp = use_temp
        self.subsample = subsample
        self.records_frac = records_frac

    def prepare_data(self) -> None:
        super().prepare_data()
        # Map codes to labels
        with open(self.classes_config, "r") as f:
            class_to_label = yaml.safe_load(f)
            self.label_to_class = {v: k for k, vs in class_to_label.items() for v in vs}

        if self.prepare:
            for stage in ["train", "eval"]:
                for f in glob(join(self.raw_data_root, stage, "labels", "*.csv")):
                    new_path = join(self.labels_roots[stage], basename(f))
                    shutil.copy(f, new_path)

    def get_dataset(
        self, features_root, labels_root: str, seasons: List[int], augment: bool = False
    ):
        indexes = pd.read_json(join(features_root, "indexes.json"))

        # Label loading and code mapping
        labels = pd.concat(
            [pd.read_csv(f, index_col=0) for f in glob(join(labels_root, "*.csv"))]
        )
        labels[LABEL_COL] = labels[LABEL_COL].map(self.label_to_class)
        labels = labels.query(f"{LABEL_COL} in {self.classes}")

        labels = labels_sample(
            labels,
            self.records_frac,
            self.subsample,
            seasons,
        )
        # Labels as hashmap
        labels_hmap = defaultdict(dict)
        for pid, s, l in labels[[POINT_ID_COL, SEASON_COL, LABEL_COL]].values:
            labels_hmap[pid][s] = l

        indexes = indexes[indexes[POINT_ID_COL].isin(labels_hmap)]
        temperatures = join(features_root, "temperatures") if self.use_temp else None

        return ChunkLabeledDataset(
            features_root=features_root,
            labels=labels_hmap,
            indexes=indexes,
            temperatures_root=temperatures,
            classes=self.classes,
            augment=augment,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.get_dataset(
                self.features_roots["train"],
                self.labels_roots["train"],
                self.seasons["train"],
                augment=True,
            )
            self.val_dataset = self.get_dataset(
                self.features_roots["eval"],
                self.labels_roots["eval"],
                self.seasons["eval"],
            )
        else:
            raise NotImplementedError("No implementation for stage {stage}")


class MaskedDataModule(CSVDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        prepare: bool = False,
        num_workers: int = 3,
        sample: float = 1.0,
        ablation: float = 0.15,
        raw_data_root: str = os.environ.get("RAW_DATA_ROOT", ""),
    ):
        super().__init__(
            train_features_root=join(data_root, "train", "features"),
            val_features_root=join(data_root, "eval", "features"),
            batch_size=batch_size,
            prepare=prepare,
            num_workers=num_workers,
            raw_data_root=raw_data_root,
        )

        self.ablation = ablation
        self.sample = sample

    def get_dataset(self, features_root):
        indexes = pd.read_json(join(features_root, "indexes.json")).sample(
            frac=self.sample
        )
        return ChunkMaskedDataset(
            features_root=features_root,
            indexes=indexes,
            ablation=self.ablation,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.get_dataset(self.features_roots["train"])
            self.val_dataset = self.get_dataset(self.features_roots["eval"])
        else:
            raise NotImplementedError("No implementation for stage {stage}")


class SQLDataModule(L.LightningDataModule):
    def __init__(
        self,
        database_url: str,
        classes: List[str],
        train_seasons: List[int] = [2017, 2018, 2019, 2020],
        val_seasons: List[int] = [2021],
        classes_config: str = "configs/rpg_codes.yml",
        sample: bool = True,
        fraction: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 3,
        chunk_size: int = 10,
    ):
        L.LightningDataModule.__init__(self)

        # Data
        self.classes = classes
        self.classes_config = classes_config
        self.train_seasons = train_seasons
        self.val_seasons = val_seasons

        # Hyperparams
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size

        # Subsampling
        self.sample = sample
        self.fraction = fraction

        # SQL
        self.database_url = database_url
        self.engine = create_engine(database_url)

    def prepare_data(self) -> None:
        super().prepare_data()
        # Map codes to labels
        with open(self.classes_config, "r") as f:
            class_to_label = yaml.safe_load(f)
            self.label_to_class = {v: k for k, vs in class_to_label.items() for v in vs}

    def setup(self, stage: str):
        if stage == "fit":
            labels = pd.read_sql("labels", self.engine)
            labels[LABEL_COL] = labels[LABEL_COL].map(self.label_to_class)
            labels = labels.query(f"{LABEL_COL} in {self.classes}")
            train_labels = labels_sample(
                labels.query("stage == 'train'"),
                self.fraction,
                self.sample,
                self.train_seasons,
            ).to_dict(orient="records")
            val_labels = labels_sample(
                labels.query("stage == 'val'"),
                self.fraction,
                self.sample,
                self.val_seasons,
            ).to_dict(orient="records")

            self.train_dataset = SQLDataset(
                db_url=self.database_url,
                features="points",
                labels=train_labels,
                seasons=self.train_seasons,
                classes=self.classes,
                augment=True,
                chunk_size=self.chunk_size,
            )

            self.val_dataset = SQLDataset(
                db_url=self.database_url,
                features="points",
                labels=val_labels,
                seasons=self.val_seasons,
                classes=self.classes,
                chunk_size=self.chunk_size,
            )

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


class SITSDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        classes: List[str],
        train_seasons: List[int] = [2017, 2018, 2019, 2020],
        val_seasons: List[int] = [2021],
        include_temperatures: bool = False,
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
        self.include_temperatures = include_temperatures

        # Hyperparams
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size

        # Subsampling
        self.fraction = fraction

    def get_dataset(self, stage: str, seasons: List[int]):
        # Gather
        features_file = join(self.root, stage, "features.parquet")
        temperatures_file = (
            join(self.root, stage, "temperatures.parquet")
            if self.include_temperatures
            else None
        )
        labels = pd.read_csv(
            os.path.join(self.root, stage, "labels.csv"), engine="pyarrow"
        )

        # Filter
        labels = labels_sample(labels, seasons=seasons, fraction=self.fraction)

        return SITSDataset(
            features_file,
            labels,
            self.classes,
            temperatures_file,
            augment=stage == "train",
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.get_dataset("train", self.train_seasons)
            self.val_dataset = self.get_dataset("val", self.train_seasons)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
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
