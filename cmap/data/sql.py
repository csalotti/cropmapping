from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset

from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from cmap.data.transforms import ts_transforms

from cmap.utils.constants import ALL_BANDS, POINT_ID_COL, DATE_COL, TEMP_COL


class SQLDataset(Dataset):
    def __init__(
        self,
        db_url: str,
        features: str,
        labels: List[Dict[str, Union[str, int]]],
        classes: List[str],
        temperatures: Optional[str] = None,
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
        self.labels = labels
        self.classes = { cn : i for i, cn in enumerate(classes)}

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

        # SQL
        self.db_url = db_url

    def get_connection(self):
        return create_engine(self.db_url, poolclass=NullPool).connect()

    def get_sequence(
        self, table: str, poi_id: str, season: int, connection
    ) -> pd.DataFrame:
        query = (
            f"SELECT * from {table} "
            + f"WHERE {POINT_ID_COL} = '{poi_id}'"
            + f" AND {DATE_COL} >= '{season - 1}-{self.start_month}-01'"
            + f" AND {DATE_COL} <= '{season}-{self.end_month}-01'"
        )

        return pd.read_sql(query, connection)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Split data among workers

        poi_id, season, label = [
            self.labels[idx][k] for k in ["poi_id", "season", "label"]
        ]

        with self.get_connection() as connection:
            season_features_df = self.get_sequence(
                self.features, poi_id, season, connection
            )

            ts = season_features_df[ALL_BANDS].values
            dates = season_features_df[DATE_COL]
            temperatures = None

            # Augment with temperatures
            if self.temperatures:
                temperatures = self.get_sequence(
                    self.temperatures, poi_id, season, connection
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

            return tensor_output
