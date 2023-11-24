import logging
from datetime import datetime
from os.path import join
from pprint import pformat
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from pandas.io.parsers.readers import TextFileReader
from torch.utils.data import IterableDataset

from utils.constants import (
    ALL_BANDS,
    CHUNK_ID_COL,
    DATE_COL,
    LABEL_COL,
    POINT_ID_COL,
    SEASON_COL,
    SIZE_COL,
    START_COL,
)


logger = logging.getLogger("lightning.pytorch.data.ChunkDataset")
# logger.addHandler(logging.FileHandler("dataset.log"))
REFERENCE_YEAR = 2023
MIN_DAYS = 3


class ChunkDataset(IterableDataset):
    def __init__(
        self,
        features_root: str,
        labels: pd.DataFrame,
        indexes: pd.DataFrame,
        classes: List[str],
        label_to_class: Dict[str, int],
        start_month: int = 11,
        end_month: int = 12,
        n_steps: int = 3,
        standardize: bool = False,
        max_records: int = -1,
    ):
        self.features_root = features_root
        self.labels = labels
        self.indexes = indexes.sort_values([CHUNK_ID_COL, START_COL])
        self.classes = {cn: i for i, cn in enumerate(classes)}
        self.label_to_class = label_to_class
        self.start_month = start_month
        self.end_month = end_month
        self.max_n_days = (
            (
                datetime(year=REFERENCE_YEAR, month=end_month, day=1)
                - datetime(year=REFERENCE_YEAR - 1, month=start_month, day=1)
            ).days
            + 1
        ) // n_steps
        self.standardize = standardize
        self.max_records = max_records

        logger.debug(f"Time Series sampled on {self.max_n_days} days")

    def transforms(self, season_features_df: pd.DataFrame, season: int):
        days = season_features_df[DATE_COL].copy()  # T
        ts = season_features_df[ALL_BANDS].values.astype(np.float32)  # T, B

        # Bands standardization
        if self.standardize:
            ts -= np.mean(ts, axis=0)
            ts /= np.std(ts, axis=0)
        else:
            ts /= 10_000

        # Days normalizatioin to ref date
        days_norm = (
            days - datetime(year=season - 1, month=self.start_month, day=1)
        ).dt.days

        # Constant padding to fit fixed size
        n_days = len(days_norm)
        ts_padded = np.pad(
            ts,
            np.array([(self.max_n_days - n_days, 0), (0, 0)]),
            constant_values=0,
        )
        days_padded = np.pad(
            days_norm,
            (self.max_n_days - n_days, 0),
            constant_values=0,
        )
        mask = np.array([False] * (self.max_n_days - n_days) + [True] * n_days)

        # Add fourth and fifth dimension for conv
        ts_expanded = np.expand_dims(np.expand_dims(ts_padded, axis=-1), axis=-1)

        return ts_expanded, days_padded, mask

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

        sub_labels_df = self.labels[
            self.labels[POINT_ID_COL].isin(sub_indexes_df[POINT_ID_COL])
        ]

        workers_chunks = sub_indexes_df[CHUNK_ID_COL].unique()
        np.random.shuffle(workers_chunks)

        for chunk_id in workers_chunks:
            chunk_reader = self._read_chunk(chunk_id)
            chunk_records = sub_indexes_df.query(f"{CHUNK_ID_COL} == {chunk_id}")
            cur_row = 0

            # Iterate over records
            for record_idx in chunk_records.to_dict(orient="records"):
                logger.debug(f"Record : {pformat(record_idx)}")
                logger.debug(f"Last row id {cur_row}")

                # Get single point time series data
                record_start_row = record_idx[START_COL]
                chunk_size = record_idx[SIZE_COL]
                if record_start_row != cur_row:
                    chunk_reader.get_chunk(record_start_row - cur_row)

                records_features_df = chunk_reader.get_chunk(chunk_size + 1)
                cur_row = record_start_row + chunk_size + 1

                # Invalid record
                if record_idx[SIZE_COL] <= MIN_DAYS:
                    continue

                if len(records_features_df[POINT_ID_COL].unique()) > 1:
                    raise ValueError(
                        f"Chunk size {record_idx[SIZE_COL]} sampled two points {records_features_df[POINT_ID_COL].unique()}\n{records_features_df}"
                    )

                records_label_df = sub_labels_df.query(
                    f"{POINT_ID_COL} == '{record_idx[POINT_ID_COL]}'"
                )

                # Produce single data bundle per season
                for _, (season, label) in (
                    records_label_df[[SEASON_COL, LABEL_COL]].sample(frac=1).iterrows()
                ):
                    logger.debug(f"Saison {season} - Label {label}")

                    season_features_df = records_features_df.query(
                        f"(({DATE_COL}.dt.year == {season - 1}) and ({DATE_COL}.dt.month >= {self.start_month})) "
                        + f" or (({DATE_COL}.dt.year == {season}) and ({DATE_COL}.dt.month < {self.end_month}))"
                    ).sort_values(DATE_COL)

                    class_id = np.array([self.classes[label]])

                    ts, days, mask = self.transforms(season_features_df, season)

                    logger.debug(
                        f"Shapes :\n\tdays\t: {days.shape}"
                        + f"\n\tmask\t: {mask.shape}"
                        + f"\n\tts\t: {ts.shape}"
                        + f"\n\tclass_id: {class_id.shape}"
                    )

                    output = {
                        "days": days,
                        "mask": mask,
                        "ts": ts,
                        "class": class_id,
                    }

                    logger.debug(output)

                    tensor_output = {
                        key: torch.from_numpy(value) for key, value in output.items()
                    }

                    yield tensor_output
