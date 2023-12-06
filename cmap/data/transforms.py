import numpy as np
import pandas as pd
from datetime import datetime
from numpy.typing import NDArray

from cmap.utils.constants import LABEL_COL, SEASON_COL


def labels_subsampling(labels: pd.DataFrame, frac: float):
    labels_dist = labels[LABEL_COL].value_counts().reset_index()
    if labels_dist.iloc[0][LABEL_COL] == "other":
        n_samples = int(1.01 * labels_dist.iloc[1, 1])
        labels = pd.concat(
            [
                labels.query(f"{LABEL_COL} == 'other'").sample(n_samples),
                labels.query(f"{LABEL_COL} != 'other'"),
            ]
        )

    # Subsample dataset respecting distribution of classes
    if frac < 1.0:
        labels = labels.groupby([LABEL_COL, SEASON_COL], group_keys=False).apply(
            lambda x: x.sample(frac=frac)
        )

    return labels


def ts_transforms(
    ts: NDArray,
    dates: pd.Series,
    season: int,
    temperatures: NDArray,
    start_month: int = 11,
    max_n_positions: int = 397,
    standardize: bool = False,
    augment: bool = False,
):
    ts = ts.astype(np.float32)

    # Bands standardization
    if standardize:
        ts -= np.mean(ts, axis=0)
        ts /= np.std(ts, axis=0)
    else:
        ts /= 10_000.0

    # data augmentation
    if augment:
        sigma = 1e-2
        clip = 3e-2
        ts = (
            ts + np.clip(np.random.normal(0, sigma, size=ts.shape), -1 * clip, clip)
        ).astype(np.float32)

    # Days normalizatioin to ref date
    days = (dates - datetime(year=season - 1, month=start_month, day=1)).dt.days.values

    # GDD computation
    if temperatures is not None:
        temperatures = np.cumsum(temperatures)[days]

    # Positions
    positions = temperatures if temperatures is not None else days

    # Constant padding to fit fixed size
    n_positions = len(positions)
    ts_padded = np.pad(
        ts,
        np.array([(0, max_n_positions - n_positions), (0, 0)]),
        constant_values=0,
    )
    positions_padded = np.pad(
        np.array(positions),
        (0, max_n_positions - n_positions),
        constant_values=0,
    )
    days_padded = np.pad(
        days,
        (0, max_n_positions - n_positions),
        constant_values=0,
    )
    mask = np.zeros(max_n_positions, dtype="uint8")
    mask[:n_positions] = 1

    return (
        ts_padded,
        positions_padded,
        days_padded,
        mask,
    )
