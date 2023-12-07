import logging
from datetime import datetime
from os.path import join
from pprint import pformat
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from pandas.io.parsers.readers import TextFileReader
from torch.utils.data import IterableDataset

from cmap.data.transforms import ts_transforms
from cmap.utils.constants import (
    ALL_BANDS,
    CHUNK_ID_COL,
    DATE_COL,
    POINT_ID_COL,
    TEMP_COL,
    SIZE_COL,
    START_COL,
)

logger = logging.getLogger("cmap.data.ChunkDataset")
# logger.addHandler(logging.FileHandler("dataset.log"))
REFERENCE_YEAR = 2023
MIN_DAYS = 10


class ChunkDataset(IterableDataset):
    def __init__(
        self,
        features_root: str,
        indexes: pd.DataFrame,
        seasons: Dict[str, List[int]],
        temperatures_root: Optional[str] = None,
        start_month: int = 11,
        end_month: int = 12,
        n_steps: int = 3,
        chunk_size: int = 10,
        standardize: bool = False,
        augment: bool = False,
    ):
        self.features_root = features_root
        self.indexes = indexes.sort_values([CHUNK_ID_COL, START_COL])
        self.seasons = seasons
        self.temperatures_root = temperatures_root
        self.start_month = start_month
        self.end_month = end_month
        self.max_n_positions = (
            (
                datetime(year=REFERENCE_YEAR, month=end_month, day=1)
                - datetime(year=REFERENCE_YEAR - 1, month=start_month, day=1)
            ).days
            + 1
        ) // n_steps
        self.chunk_size = chunk_size
        self.standardize = standardize
        self.augment = augment

    def _read_chunk(self, chunk_id: int) -> TextFileReader:
        date_col_idx = (
            open(join(self.features_root, f"chunk_{chunk_id}.csv"), "r")
            .readline()
            .split(",")
            .index(DATE_COL)
        )
        chunk_it = pd.read_csv(
            join(self.features_root, f"chunk_{chunk_id}.csv"),
            iterator=True,
            parse_dates=[date_col_idx],
        )
        return chunk_it

    def __season_filter(self, season: int):
        return (
            f"(({DATE_COL}.dt.year == {season - 1})"
            + f" and ({DATE_COL}.dt.month >= {self.start_month}))"
            + f" or (({DATE_COL}.dt.year == {season})"
            + f" and ({DATE_COL}.dt.month < {self.end_month}))"
        )

    def __iter__(self):
        # Workers infos
        worker_info = torch.utils.data.get_worker_info()

        # Split data among workers
        sub_indexes_df = (
            self.indexes.query(
                f"{CHUNK_ID_COL} % {worker_info.num_workers} == {worker_info.id}"
            )
            if worker_info.num_workers > 1
            else self.indexes
        )

        workers_chunks = sub_indexes_df[CHUNK_ID_COL].unique()

        for chunk_id in workers_chunks:
            chunk_reader = self._read_chunk(chunk_id)
            chunk_records = sub_indexes_df.query(f"{CHUNK_ID_COL} == {chunk_id}")
            cur_row = 0

            chunk_poi_ids = chunk_records[POINT_ID_COL].values
            for i in range(0, len(chunk_poi_ids) // self.chunk_size):
                sub_chunk_ids = chunk_poi_ids[
                    (i * self.chunk_size) : min(
                        (i + 1) * self.chunk_size,
                        len(chunk_poi_ids),
                    )
                ].tolist()

                sub_chunk_temps = None
                if self.temperatures_root:
                    sub_chunk_temps = pd.concat(
                        [
                            pd.read_csv(
                                join(self.temperatures_root, f"{id}.csv"),
                                parse_dates=[DATE_COL],
                            )
                            for id in sub_chunk_ids
                        ],
                        ignore_index=True,
                    ).sort_values([DATE_COL])

                # Iterate over records
                sub_chunk_records = chunk_records.query(
                    f"{POINT_ID_COL} in @sub_chunk_ids"
                ).to_dict(orient="records")

                for record_idx in sub_chunk_records:
                    # Get single point time series data
                    poi_id = record_idx[POINT_ID_COL]
                    record_start_row = record_idx[START_COL]
                    chunk_size = record_idx[SIZE_COL]
                    if record_start_row != cur_row:
                        chunk_reader.get_chunk(record_start_row - cur_row)

                    records_features_df = chunk_reader.get_chunk(
                        chunk_size + 1
                    ).sort_values(DATE_COL)
                    records_features_df.columns = (
                        records_features_df.columns.str.strip().str.lower()
                    )
                    cur_row = record_start_row + chunk_size + 1

                    if len(records_features_df[POINT_ID_COL].unique()) > 1:
                        raise ValueError(
                            f"Chunk size {record_idx[SIZE_COL]} sampled two points"
                            + f"{records_features_df[POINT_ID_COL].unique()}\n{records_features_df}"
                        )

                    record_temperatures = None
                    if sub_chunk_temps is not None:
                        record_temperatures = sub_chunk_temps.query(
                            f"{POINT_ID_COL} == '{poi_id}'"
                        )

                    for season in self.seasons[poi_id]:
                        season_features_df = records_features_df.query(
                            self.__season_filter(season)
                        )

                        # Invalid record
                        if len(season_features_df) <= MIN_DAYS:
                            continue

                        ts = season_features_df[ALL_BANDS].values
                        dates = season_features_df[DATE_COL]
                        temperatures = None

                        # Augment with temperatures
                        if record_temperatures is not None:
                            temperatures = record_temperatures.query(
                                self.__season_filter(season)
                            ).sort_values(DATE_COL)

                            temperatures = temperatures[TEMP_COL].values

                        ts, positions, days, mask = ts_transforms(
                            ts=ts,
                            dates=dates,
                            temperatures=temperatures,
                            season=season,
                            start_month=self.start_month,
                            max_n_positions=self.max_n_positions,
                            standardize=self.standardize,
                            augment=self.augment,
                        )

                        yield poi_id, season, ts, positions, days, mask


class ChunkLabeledDataset(ChunkDataset):
    def __init__(
        self,
        features_root: str,
        labels: Dict[str, Dict[int, str]],
        indexes: pd.DataFrame,
        classes: List[str],
        temperatures_root: Optional[str] = None,
        start_month: int = 11,
        end_month: int = 12,
        n_steps: int = 3,
        standardize: bool = False,
        augment: bool = False,
    ):
        super().__init__(
            features_root=features_root,
            indexes=indexes,
            temperatures_root=temperatures_root,
            seasons={k: list(v.keys()) for k, v in labels.items()},
            start_month=start_month,
            end_month=end_month,
            n_steps=n_steps,
            standardize=standardize,
            augment=augment,
        )
        self.labels = labels
        self.classes = {cn: i for i, cn in enumerate(classes)}

    def __iter__(self):
        for poi_id, season, ts, positions, days, mask in super().__iter__():
            label = self.labels[poi_id].get(season)
            if label is not None:
                class_id = np.array([self.classes[label]])

                output = {
                    "positions": positions,
                    "days": days,
                    "mask": mask,
                    "ts": ts,
                    "class": class_id,
                }

                tensor_output = {
                    key: torch.from_numpy(value) for key, value in output.items()
                }

                yield tensor_output


class ChunkMaskedDataset(ChunkDataset):
    def __init__(
        self,
        features_root: str,
        indexes: pd.DataFrame,
        temperatures_root: Optional[str] = None,
        n_bands: int = 9,
        start_month: int = 11,
        end_month: int = 12,
        n_steps: int = 3,
        standardize: bool = False,
        ablation: float = 0.15,
    ):
        super().__init__(
            features_root=features_root,
            indexes=indexes,
            temperatures_root=temperatures_root,
            start_month=start_month,
            end_month=end_month,
            n_steps=n_steps,
            standardize=standardize,
        )

        self.ablation = ablation
        self.mask = np.random.normal(0, 1e-2, size=n_bands)

    def random_masking(self, ts, ts_length):
        ts_masked = ts.copy()
        days_masked = np.array([0] * self.max_n_positions)
        n_removed_days = int(self.ablation * ts_length)

        # if random() < 0.5:
        #     start_idx = np.random.choice(range(ts_length - n_removed_days), 1)[0]
        #     days_masked[start_idx : start_idx + n_removed_days] = 1
        #     ts_masked[start_idx : start_idx + n_removed_days] = self.mask
        # else:
        indexes = np.random.choice(
            range(ts_length),
            n_removed_days,
        )
        days_masked[indexes] = 1
        ts_masked[indexes] = self.mask

        return ts_masked, days_masked

    def __iter__(self):
        for _, season, ts, positions, days, mask in super().__iter__():
            # Random masking of time stamps
            ts_masked, loss_mask = self.random_masking(ts, sum(mask))

            output = {
                "days": days,
                "positions": positions,
                "ts": ts_masked,
                "target": ts,
                "mask": mask,
                "loss_mask": loss_mask,
                "season": np.array([season]),
            }

            tensor_output = {
                key: torch.from_numpy(value) for key, value in output.items()
            }

            yield tensor_output
