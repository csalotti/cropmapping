from collections import defaultdict
from datetime import datetime
from typing import Dict, List
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
    """Satelite Image Time Serie IterableDataset

    Provide iteration capabilities over time series features
    and extra ones, such as temperatures and precipitations

    Attributes:
        features_file (str) : path to features parquet
        labels (Dict[str, Dict[str]]) : Registry of label per points per seasons.
            Format :
            { '<point_id> : { <season_1> : <label_1>, <season_2> : <label_2>}}
        classes (List[str]) : Sets of expected classes
        extra_features_files (Dict[str, Dict[str, str]]) : Maps of additionnal files with
            their name and path for train and validation datasets.
            The dictionnary must have the following structure:
            {'temperatures' : {'train' : <path>, 'val' : <path}}
        start_month (int) : Starting season month for season - 1
        end_month (int) : Ending season month for season (can overlap with season + 1)
        max_n_positions (int) : Maximum number of positions sampled on time serie
        standardize (bool): Flag for ts values standardization ( x - mean) / std,
            otherwise, divide by 10_000
        augment (bool) : Add N(0,10-2) noise to each bands

    """

    def __init__(
        self,
        features_file: str,
        labels: pd.DataFrame,
        classes: List[str],
        extra_features_files: Dict[str, str] = {},
        ref_year: int = 2023,
        start_month: int = 11,
        end_month: int = 12,
        n_steps: int = 3,
        standardize: bool = False,
        augment: bool = False,
    ):
        """Initialize Dataset with a data attributees, seasons steps and processing
        options

        Args:
            features_file (str) : path to features parquet
            labels (pd.DataFrame): Labels DataFrame
            classes (List[str]): List of classes
            extra_features_files (Dict[str, Dict[str, str]]) : Maps of additionnal
                files with their name and path for train and validation datasets.
                The dictionnary must have the following structure:
                {'temperatures' : {'train' : <path>, 'val' : <path}}
            ref_year (int): Year to compute number of maximum positions.
            start_month (int): Starting season month for season - 1
            end_month (int): Ending season month for season (can overlap with season + 1)
            n_steps (int): Number of step for maximmum positions
            standardize (bool): Flag for ts values standardization ( x - mean) / std,
                otherwise, divide by 10_000
            augment (bool): Add N(0,10-2) noise to each bands
        """
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

    def get_table(self, file: str, poi_ids: List[str]) -> pd.DataFrame:
        """Retrieve row from parquet given a set of point ids

        Args:
            file (str): parquet file path
            poi_ids (List[str]): Selected point idsa

        Returns:
            Dataframe with point ids data

        """
        with open(file, "rb") as f:
            df = pd.read_parquet(f, filters=[(POINT_ID_COL, "in", poi_ids)])
        return df

    def filter_season(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Filter time series given the season. It is based
        on start and end month definition

        Args:
            df (pd.DataFrame): Dataframe to filter
            season (int): Season to select

        Returns:
            Filtered time serie dataframe
        """
        return df.query(
            f"({DATE_COL} >= '{season - 1}-{self.start_month}-01')"
            + f" & ({DATE_COL} < '{season}-{self.end_month}-01')"
        )

    def __iter__(self):
        """Main Dataset iteration loop

        Due to duplication by the Dataloader, data is split evenly
        among workers (same number of points per workers)
        """
        # Workers infos split
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

        # Workers features attribution
        worker_features_df = self.get_table(self.features_file, worker_poi_ids)
        worker_features_df.columns = worker_features_df.columns.str.strip().str.lower()
        iterator = [worker_features_df.groupby(POINT_ID_COL)]

        # Extra features augmentation
        extra_features_name = []
        for extra_id, extra_file in self.extra_features_files:
            extra_features_name.append(extra_id)
            worker_extra_df = self.get_table(extra_file, worker_poi_ids)
            iterator.append(worker_extra_df.groupby(POINT_ID_COL))

        iterator = zip(*iterator) if len(extra_features_name) > 0 else iterator[0]

        # Iteration through group per point id.
        # If additionnal features are added, they will be included
        # in the loop
        for group_features in iterator:
            poi_id, features_df = (
                group_features if len(extra_features_name) == 0 else group_features[0]
            )

            # Season filtering
            for season in self.labels[poi_id].keys():
                season_features_df = self.filter_season(features_df, season)

                if len(season_features_df) < 5:
                    continue

                extra_features_df = {}

                # Extra features season filtering
                for i, extra_name in enumerate(extra_features_name):
                    if group_features[i + 1][0] != poi_id:
                        ValueError(
                            f"{extra_name} and features don't match"
                            + f"{group_features[i+1][0]} != {poi_id}"
                        )

                    extra_features_df[extra_name] = self.filter_season(
                        group_features[i + 1][1], season
                    )

                ts = season_features_df[ALL_BANDS].values
                dates = season_features_df[DATE_COL].dt.date.values
                label = self.labels[poi_id][season]

                # Features transforms
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

                # Convert to dicto of tensors
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
