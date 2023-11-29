import json
import logging
from glob import glob
from os.path import join
from typing import List

import pandas as pd
import pytorch_lightning as L
import multiprocessing
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

from cmap.data.dataset import ChunkLabeledDataset, ChunkMaskedDataset
import os
from glob import glob
from cmap.utils.chunk import chunks_indexing, preprocessing
from cmap.utils.constants import LABEL_COL, POINT_ID_COL, SEASON_COL

logger = logging.getLogger("cmap.data.module")
# logger.addHandler(logging.FileHandler("datamodule.log"))


class SITSDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_features_root: str,
        val_features_root: str,
        batch_size: int = 32,
        prepare: bool = False,
        num_workers: int = 3,
        raw_data_root: str = os.environ.get("RAW_DATA_ROOT", ""),
        include_temp: bool = False,
    ):
        L.LightningDataModule.__init__(self)

        self.train_features_root = train_features_root
        self.val_features_root = val_features_root
        self.batch_size = batch_size
        self.prepare = prepare
        self.raw_data_root = raw_data_root
        self.num_workers = num_workers
        self.include_temp = include_temp

    def _prepare_chunk(self, chunk_id):
        stage = "train" if chunk_id <= 80 else "eval"
        features_df = pd.read_csv(
            join(self.raw_data_root, stage, "features", f"chunk_{chunk_id}")
        )
        temp_files = []
        if self.include_temp:
            temp_files = glob(
                join(
                    self.raw_data_root,
                    "temperatures",
                    chunk_id,
                    "*.csv",
                )
            )

        indexes, features_df = preprocessing(features_df, temp_files)

        features_df.to_csv(
            join(self.train_features_root, "features", f"chunk_{chunk_id}")
        )
        return indexes

    def prepare_data(self) -> None:
        if self.prepare:
            with multiprocessing.Pool() as pool:
                indexes_map = pool.map(self._prepare_chunk, range(100))

            train_indexes = []
            val_indexes = []
            for i, ind in enumerate(indexes_map):
                if i <= 80:
                    train_indexes.extend(ind)
                else:
                    val_indexes.extend(ind)

            with open(join(self.train_features_root, "indexes.json"), "r") as f:
                json.dump(train_indexes, f)

            with open(join(self.val_features_root, "indexes.json"), "r") as f:
                json.dump(val_indexes, f)

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
            pin_memory=self.trainer.num_devices > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=self.trainer.num_devices > 0,
        )


class LabelledDataModule(SITSDataModule):
    def __init__(
        self,
        data_root: str,
        classes: List[str],
        classes_config: str = "configs/rpg_codes.yml",
        batch_size: int = 32,
        prepare: bool = False,
        num_workers: int = 3,
        records_frac: float = 1,
        subsample: bool = False,
    ):
        super().__init__(
            join(data_root, "train", "features"),
            join(data_root, "eval", "features"),
            batch_size,
            prepare,
            num_workers,
        )

        self.train_label_root = join(data_root, "train", "label")
        self.val_label_root = join(data_root, "val", "label")
        self.classes = classes
        self.classes_config = classes_config
        self.subsample = subsample
        self.records_frac = records_frac

    def prepare_data(self) -> None:
        super().prepare_data()
        # Map codes to labels
        with open(self.classes_config, "r") as f:
            class_to_label = yaml.safe_load(f)
            self.label_to_class = {v: k for k, vs in class_to_label.items() for v in vs}

    def get_dataset(self, features_root, labels_root: str):
        indexes = pd.read_json(join(features_root, "indexes.json"))

        # Label loading and code mapping
        labels = pd.concat(
            [pd.read_csv(f, index_col=0) for f in glob(join(labels_root, "*.csv"))]
        )
        labels[LABEL_COL] = labels[LABEL_COL].map(
            lambda x: self.label_to_class.get(x, "other")
        )
        labels = labels.query(f"{LABEL_COL} in {self.classes}")

        # Subsample other class to be 1% highe than second top
        if self.subsample:
            labels_dist = labels[LABEL_COL].value_counts().reset_index()
            if labels_dist.iloc[0][LABEL_COL] == "other":
                n_samples = int(1.01 * labels_dist.iloc[1, 1])
                labels = pd.concat(
                    [
                        labels.query(f"{LABEL_COL} == 'other'").sample(n_samples),
                        labels.query(f"{LABEL_COL} != 'other'"),
                    ]
                )

            # Subsample dataset respecting distribution of classes
            if self.records_frac < 1.0:
                labels = labels.groupby(
                    [LABEL_COL, SEASON_COL], group_keys=False
                ).apply(lambda x: x.sample(frac=self.records_frac))

        indexes = indexes[indexes[POINT_ID_COL].isin(labels[POINT_ID_COL])]

        return ChunkLabeledDataset(
            features_root=features_root,
            labels=labels,
            indexes=indexes,
            classes=self.classes,
            label_to_class=self.label_to_class,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.get_dataset(
                self.train_features_root,
                self.train_label_root,
            )
            self.val_dataset = self.get_dataset(
                self.val_features_root,
                self.val_label_root,
            )
        else:
            raise NotImplementedError("No implementation for stage {stage}")


class MaskedDataModule(SITSDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        prepare: bool = False,
        num_workers: int = 3,
        sample: float = 1.0,
        ablation: float = 0.15,
    ):
        super().__init__(
            join(data_root, "train", "features"),
            join(data_root, "eval", "features"),
            batch_size,
            prepare,
            num_workers,
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
            self.train_dataset = self.get_dataset(self.train_features_root)
            self.val_dataset = self.get_dataset(self.val_features_root)
        else:
            raise NotImplementedError("No implementation for stage {stage}")
