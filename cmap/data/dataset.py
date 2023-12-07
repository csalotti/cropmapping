from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from cmap.data.transforms import ts_transforms

from cmap.utils.constants import (
    ALL_BANDS,
    DATE_COL,
    POINT_ID_COL,
    TEMP_COL,
)


class SITSDataset(Dataset):
    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        classes: List[str],
        temperatures: Optional[pd.DataFrame] = None,
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
        self.classes = {cn: i for i, cn in enumerate(classes)}

        # Labels as hashmap
        self.labels = labels.to_dict(orient="records")

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

    def filter_season(self, df: pd.DataFrame, poi_id: str, season: int) -> pd.DataFrame:
        return df.query(
            f"{POINT_ID_COL} == '{poi_id}'"
            + f"(date >= '{season - 1}-{self.start_month}-01')"
            + f" & (date <= '{season}-{self.end_month}-01')"
        )

    def __getitem__(self, idx):
        poi_id, season, label = [
            self.labels[idx][k] for k in ["poi_id", "season", "label"]
        ]

        season_features_df = self.filter_season(self.features, poi_id, season)
        season_features_df.columns = season_features_df.columns.str.strip().str.lower()
        ts = season_features_df[ALL_BANDS].values
        dates = season_features_df[DATE_COL]
        temperatures = None

        # Augment with temperatures
        if self.temperatures is not None:
            temperatures = self.temperatures.query(f"{POINT_ID_COL} == {poi_id}")
            temperatures = self.filter_season(temperatures, poi_id, season)
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

        tensor_output = {key: torch.from_numpy(value) for key, value in output.items()}

        return tensor_output
