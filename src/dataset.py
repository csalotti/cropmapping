from os.path import join
import pandas as pd
import numpy as np
from datetime import datetime
import torch
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

import logging

CLASS_TO_LABEL = {
    1: ["AVH", "AVP"],  # avoine
    2: ["BTH", "BTP"],  # ble_tendre
    3: ["BDH", "BDP", "BDT"],  # ble_dur
    4: ["BTN", "BVF"],  # betterave
    5: ["CZH", "CZP"],  # colza
    6: ["FVL", "FVT", "FFO", "FF5", "FF6", "FF7", "DFV"],  # feveroles
    7: ["HAR"],  # haricots
    8: ["LEC", "DLL", "LEF"],  # lentilles
    9: ["LIH", "LIP", "LIF", "DLN"],  # lin
    10: [
        "LDP",
        "LDT",
        "LDH",
        "LFH",
        "LFP",
        "LH5",
        "LH6",
        "LH7",
        "LP5",
        "LP6",
        "LP7N",
        "DLP",
    ],  # lupin
    11: ["MID", "MIE", "MIS"],  # mais
    12: ["MLT", "DML"],  # millet
    13: ["ORH", "ORP"],  # orge
    14: [
        "PHI",
        "PPR",
        "PPT",
        "PFH",
        "PFP",
        "PH5",
        "PH6",
        "PH7",
        "PP5",
        "PP6",
        "PP7",
        "DPS",
    ],  # pois
    15: ["SGH", "SGP", "DSG"],  # seigle
    16: ["SOJ", "DSJ"],  # soja
    17: ["SOG", "DSF"],  # sorgho
    18: ["TRN", "DTN"],  # tournesol
    19: ["TTH", "TTP"],  # triticale
}

logger = logging.getLogger(__name__)

REFERENCE_YEAR = 2023


class ChunkDataset(IterableDataset):
    def __init__(
        self,
        features_root: str,
        labels_root: str,
        indexes: pd.DataFrame,
        start_month: int = 11,
        end_month: int = 12,
        n_steps: int = 3,
        standardize: bool = False,
    ):
        self.features_root = features_root
        self.labels_root = labels_root
        self.indexes = iter(indexes.to_dict(orient="records"))
        self.label_to_class = {
            class_id: label_name
            for class_id, labels_names in CLASS_TO_LABEL.items()
            for label_name in labels_names
        }
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

        logger.debug(f"Time Series sampled on {self.max_n_days} days")

    def transforms(self, ts, days):
        # Data standardization
        if self.standardize:
            ts -= np.mean(ts, axis=0)
            ts /= np.std(ts, axis=0)
        else:
            ts /= 10_000

        # Constant padding to fit fixed size
        n_days = len(days)
        ts_padded = np.pad(
            ts, np.array([(self.max_n_days - n_days, 0), (0, 0)]), constant_values=-100
        )
        days_padded = np.pad(days, (self.max_n_days - n_days, 0), constant_values=-100)
        mask = days_padded == -100

        return ts_padded, days_padded, mask

    def __iter__(self):
        record_idx = next(self.indexes)

        # Get single point time series data
        records_features_df = (
            pd.read_csv(
                join(self.features_root, f"chunk_{record_idx[CHUNK_ID_COL]}.csv"),
                skiprows=record_idx[START_COL],
                iterator=True,
            )
            .get_chunk(record_idx[SIZE_COL])
            .assign(**{DATE_COL: lambda x: pd.to_datetime(x[DATE_COL])})
        )

        records_label_df = pd.read_csv(
            join(self.labels_root, f"chunk_{record_idx[CHUNK_ID_COL]}.csv")
        ).query(f"{POINT_ID_COL} == '{record_idx[POINT_ID_COL]}'")

        # Produce single data bundle per season
        for _, (season, label) in (
            records_label_df[[SEASON_COL, LABEL_COL]].sample(frac=1).iterrows()
        ):
            logger.debug(f"Saison {season} - Label {label}")

            season_features_df = records_features_df.query(
                f"(({DATE_COL}.dt.year == {season - 1}) and ({DATE_COL}.dt.month >= {self.start_month})) "
                + f" or (({DATE_COL}.dt.year == {season}) and ({DATE_COL}.dt.month < {self.end_month}))"
            ).sort_values(DATE_COL)

            days = season_features_df[DATE_COL].dt.dayofyear.values
            ts = season_features_df[ALL_BANDS].values.astype(np.float32)
            class_id = np.array([self.label_to_class.get(label, 0)])

            ts, days, mask = self.transforms(ts, days)

            logger.debug(
                f"""
                Shapes : 
                    days    : {days.shape}
                    mask    : {mask.shape}
                    ts      : {ts.shape}
                    class_id: {class_id.shape}
                          """
            )

            output = {
                "days": days,
                "mask": mask,
                "ts": ts,
                "class": class_id,
            }

            yield {key: torch.from_numpy(value) for key, value in output.items()}
