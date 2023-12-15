import gc
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from cmap.data.transforms import ts_transforms
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
        extra_features_files (Dict[str, Dict[str, str | List[str]]]) : Maps of additionnal files with
            their name and path for train and validation datasets.
            The dictionnary must have the following structure:
            {'temperatures' : {'path' : <fpath> , 'features_cols' : [<f1>, <f2>]}}
        chunk_size (int) : Number of poi_id query and process at the same time in memory.
            if -1, all the workers ids are considered (default : -1)
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
        extra_features_files: Dict[str, Dict[str, str | List[str]]] = {},
        chunk_size: int = -1,
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
            extra_features_files (Dict[str, Dict[str, str | List[str]]]) : Maps of additionnal files with
                files with their name and path for train and validation datasets.
                The dictionnary must have the following structure:
                {'temperatures' : {'path' : <fpath> , 'features_cols' : [<f1>, <f2>]}}
            chunk_size (int) : Number of poi_id query and process at the same time in memory.
                if -1, all the workers ids are considered (default : -1)
            ref_year (int): Year to compute number of maximum positions (default : 2023).
            start_month (int): Starting season month for season - 1 (default : 11)
            end_month (int): Ending season month for season (can overlap with season + 1) (default : 12)
            n_steps (int): Number of step for maximmum positions (default : 5)
            standardize (bool): Flag for ts values standardization ( x - mean) / std,
                otherwise, divide by 10_000 (default : False)
            augment (bool): Add N(0,10-2) noise to each bands (default : False)
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
        self.chunk_size = chunk_size
        self.standardize = standardize
        self.augment = augment

    def get_season_table(
        self, file: str, poi_id: str, season: int, cols: List[str]
    ) -> pd.DataFrame:
        """Retrieve row from parquet given a set of point ids

        Args:
            file (str): parquet file path
            poi_id (str): Selected point id
            season (int) : Selected season
            cols (List[str]): Selected columns

        Returns:
            Dataframe with point ids data

        """
        with open(file, "rb") as f:
            df = pd.read_parquet(
                f,
                columns=cols,
                filters=[
                    (POINT_ID_COL, "=", poi_id),
                    (DATE_COL, ">=", f"{season -1}-{self.start_month}-01"),
                    (DATE_COL, "<", f"{season }-{self.end_month}-01"),
                ],
            )
        return df

    def get_table(self, file: str, poi_ids: List[str], cols: List[str]) -> pd.DataFrame:
        """Retrieve row from parquet given a set of point ids

        Args:
            file (str): parquet file path
            poi_ids (List[str]): Selected point ids
            cols (List[str]): Selected columns

        Returns:
            Dataframe with point ids data

        """
        with open(file, "rb") as f:
            df = pd.read_parquet(
                f, columns=cols, filters=[(POINT_ID_COL, "in", poi_ids)]
            )
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
        features_df = self.get_table(
            self.features_file,
            worker_poi_ids,
            cols=[POINT_ID_COL, DATE_COL] + ALL_BANDS,
        )

        # Iteration through group per point id.
        # If additionnal features are added, they will be included
        # in the loop
        for poi_id, features_df in features_df.groupby(POINT_ID_COL):
            # Season filtering
            for season in self.labels[poi_id].keys():
                season_features_df = self.filter_season(features_df, season)

                if len(season_features_df) < 5:
                    continue

                extra_features_values = {}

                # Extra features season filtering
                for extra_id, extra_conf in self.extra_features_files.items():
                    fpath = extra_conf["path"]
                    features_cols = extra_conf["features"]
                    extra_features_values[extra_id] = self.get_season_table(
                        fpath,
                        poi_id,
                        season,
                        cols=[POINT_ID_COL, DATE_COL] + features_cols,
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
                    **extra_features_values,
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
