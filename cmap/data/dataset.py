from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import numpy as np
from cmap.data.transforms import ts_transforms
import torch

from cmap.utils.constants import (
    ALL_BANDS,
    DATE_COL,
    LABEL_COL,
    POINT_ID_COL,
    SEASON_COL,
    TEMP_COL,
)


class SITSDataset(IterableDataset):
    def __init__(
        self,
        features_file: str,
        labels: pd.DataFrame,
        classes: List[str],
        temperatures_file: str,
        ref_year: int = 2023,
        start_month: int = 11,
        end_month: int = 12,
        n_steps: int = 3,
        standardize: bool = False,
        augment: bool = False,
    ):
        # Data
        self.features_file = features_file
        self.temperatures_file = temperatures_file
        self.classes = {cn: i for i, cn in enumerate(classes)}

        # Labels
        self.labels = defaultdict(dict)
        for l in labels.to_dict("records"):
            self.labels[l[POINT_ID_COL]][l[SEASON_COL]] = l[LABEL_COL]
        self.num_points = len(self.labels.keys())

        # Dates and season norm
        self.start_month = start_month
        self.end_month = end_month
        self.max_n_positions = (
            (
                datetime(year=ref_year, month=end_month, day=1)
                - datetime(year=ref_year - 1, month=start_month, day=1)
            ).days
            + 1
        ) // n_steps

        # Processing
        self.standardize = standardize
        self.augment = augment

    def get_table(
        self,
        file: str,
        min_id: str,
        max_id: str,
    ) -> pd.DataFrame:
        return pd.read_parquet(
            file,
            filters=[
                (POINT_ID_COL, "<=", min_id),
                (POINT_ID_COL, ">=", max_id),
            ],
        )

    def filter_season(self, df: pd.DataFrame, season: int):
        return df.query(
            f"({DATE_COL} >= '{season - 1}-{self.start_month}-01')"
            + f" & ({DATE_COL} <= '{season}-{self.end_month}-01')"
        )

    def __iter__(self):
        # Workers infos
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        chunk_size = self.num_points // num_workers

        worker_poi_ids = list(self.labels.keys())[
            (worker_id * chunk_size) : min(
                (worker_id + 1) * chunk_size, self.num_points
            )
        ]
        worker_features_df = self.get_table(
            file=self.features_file,
            min_id=worker_poi_ids[0],
            max_id=worker_poi_ids[-1],
        )
        worker_features_df.columns = worker_features_df.columns.str.strip().str.lower()
        worker_temperatures_df = self.get_table(
            self.temperatures_file,
            worker_poi_ids[0],
            worker_poi_ids[-1],
        )

        for (feat_poi_id, poi_features_df), (temp_poi_id, poi_temperatures_df) in zip(
            worker_features_df.groupby(POINT_ID_COL),
            worker_temperatures_df.groupby(POINT_ID_COL),
        ):
            if feat_poi_id != temp_poi_id:
                raise ValueError(
                    "temperatures and features don't match anymore {feat_id} != {temp_id}"
                )

            for season in self.labels[feat_poi_id].keys():
                ts = poi_features_df[ALL_BANDS].values
                dates = poi_features_df[DATE_COL].values
                temperatures = poi_temperatures_df[TEMP_COL].values
                label = self.labels[feat_poi_id][season]

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
