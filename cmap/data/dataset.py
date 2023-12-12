from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional
import torch
from torch.utils.data import IterableDataset
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
)


class SITSDataset(IterableDataset):
    def __init__(
        self,
        features_file: str,
        labels: pd.DataFrame,
        classes: List[str],
        extra_features_files: Dict[str, str] = {},
        ref_year: int = 2023,
        start_month: int = 11,
        end_month: int = 12,
        n_steps: int = 5,
        standardize: bool = False,
        augment: bool = False,
    ):
        # Data
        self.features_file = features_file
        self.extra_features_files = extra_features_files
        self.classes = {cn: i for i, cn in enumerate(classes)}

        # Labels
        self.labels = defaultdict(dict)
        for poi_id, season, label in labels[
            [POINT_ID_COL, SEASON_COL, LABEL_COL]
        ].values:
            self.labels[poi_id][season] = label
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

    def get_table(self, file: str, poi_ids: List[str]):
        with open(file, "rb") as f:
            df = pd.read_parquet(
                open(file, "rb"), filters=[(POINT_ID_COL, "in", poi_ids)]
            )
        return df

    def filter_season(self, df: pd.DataFrame, season: int):
        return df.query(
            f"({DATE_COL} >= '{season - 1}-{self.start_month}-01')"
            + f" & ({DATE_COL} < '{season}-{self.end_month}-01')"
        )

    def __iter__(self):
        # Workers infos
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            chunk_size = self.num_points // num_workers

            worker_poi_ids = list(self.labels.keys())[
                (worker_id * chunk_size) : min(
                    (worker_id + 1) * chunk_size, self.num_points
                )
            ]
        else:
            worker_poi_ids = list(self.labels.keys())

        worker_features_df = self.get_table(self.features_file, worker_poi_ids)
        worker_features_df.columns = worker_features_df.columns.str.strip().str.lower()
        iterator = [worker_features_df.groupby(POINT_ID_COL)]
        extra_features_name = []
        for extra_id, extra_file in self.extra_features_files:
            extra_features_name.append(extra_id)
            worker_extra_df = self.get_table(extra_file, worker_poi_ids)
            iterator.append(worker_extra_df.groupby(POINT_ID_COL))

        # for (feat_poi_id, poi_features_df), (temp_poi_id, poi_temperatures_df) in zip(
        #     worker_features_df.groupby(POINT_ID_COL),
        #     worker_temperatures_df.groupby(POINT_ID_COL),
        # ):
        iterator = zip(*iterator) if len(extra_features_name) > 0 else iterator[0]

        for group_features in iterator:
            poi_id, features_df = (
                group_features if len(extra_features_name) == 0 else group_features[0]
            )

            for season in self.labels[poi_id].keys():
                season_features_df = self.filter_season(features_df, season)

                if len(season_features_df) < 5:
                    continue

                extra_features_df = {}

                for i, extra_name in enumerate(extra_features_name):
                    if group_features[i + 1][0] != poi_id:
                        ValueError(
                            f"{extra_name} and features don't match {group_features[i+1][0]} != {poi_id}"
                        )

                    extra_features_df[extra_name] = self.filter_season(
                        group_features[i + 1][1], season
                    )

                ts = season_features_df[ALL_BANDS].values
                dates = season_features_df[DATE_COL].dt.date.values
                label = self.labels[poi_id][season]

                ts, positions, days, mask = ts_transforms(
                    ts=ts,
                    dates=dates,
                    season=season,
                    start_month=self.start_month,
                    max_n_positions=self.max_n_positions,
                    standardize=self.standardize,
                    augment=self.augment,
                    **extra_features_df,
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
