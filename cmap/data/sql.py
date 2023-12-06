from collections import defaultdict
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset

from sqlalchemy import create_engine
from cmap.data.transforms import ts_transforms

from cmap.utils.constants import (
    ALL_BANDS,
    LABEL_COL,
    POINT_ID_COL,
    DATE_COL,
    SEASON_COL,
    TEMP_COL,
)

MIN_DAYS = 10


class SQLDataset(Dataset):
    def __init__(
        self,
        db_url: str,
        features: str,
        labels: List[Dict[str, Union[str, int]]],
        seasons: List[int],
        classes: List[str],
        temperatures: Optional[str] = None,
        chunk_size: int = 10,
        ref_year: int = 2023,
        start_month: int = 11,
        end_month: int = 12,
        n_steps: int = 3,
        standardize: bool = False,
        augment: bool = False,
    ):
        # Data
        self.features = features
        self.temperatures = temperatures
        self.seasons = seasons
        self.classes = {cn: i for i, cn in enumerate(classes)}

        # Labels as hashmapa
        self.labels = defaultdict(dict)
        for l in labels:
            self.labels[l[POINT_ID_COL]][l[SEASON_COL]] = l[LABEL_COL]

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
        self.chunk_size = chunk_size

        # Processing
        self.standardize = standardize
        self.augment = augment

        # SQL
        self.sql_engine = create_engine(
            db_url,
            echo=False,
            echo_pool=False,
            pool_size=1,
            pool_pre_ping=True,
        )
        self.sql_engine.dispose(close=True)

    def get_sequence(self, table: str, points: List[str]) -> pd.DataFrame:
        pids = ",".join(map(lambda i: f"'{i}'", points))
        query = f"SELECT * from {table} " + f"WHERE {POINT_ID_COL} = ({pids})"
        return pd.read_sql(query, self.sql_engine)

    def __getitem__(self, idx):
        # Split data among workers

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        worker_pids = range(worker_id, len(self.labels), num_workers)
        worker_pids = np.array(list(self.labels.keys()))[worker_pids]

        for i in range(len(worker_pids) // self.chunk_size):
            current_ids = worker_pids[
                self.chunk_size * i : min(self.chunk_size * (i + 1), len(worker_pids))
            ]

            chunk_features_df = self.get_sequence(self.features, current_ids)

            for _, poi_df in chunk_features_df.groupby(POINT_ID_COL):
                for season in self.seasons:
                    season_features_df = poi_df.query(
                        f"date >= {season - 1}-{self.start_month}-01"
                        + f"AND date <= {season}-{self.end_month}-01"
                    )

                    if len(season_features_df) < MIN_DAYS:
                        continue

                    ts = season_features_df[ALL_BANDS].values
                    dates = season_features_df[DATE_COL]
                    temperatures = None

                    # Augment with temperatures
                    if self.temperatures:
                        temperatures = self.get_sequence(
                            self.temperatures, poi_id, season
                        )
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
