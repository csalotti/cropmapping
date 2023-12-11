import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from datetime import date
from numpy.typing import NDArray

from cmap.utils.constants import LABEL_COL, SEASON_COL, POINT_ID_COL


def labels_sample(labels: pd.DataFrame, fraction: float, seasons: List[int]):
    labels = labels.query(f"{SEASON_COL} in {seasons}")

    # Subsample dataset respecting distribution of classes
    if fraction < 1.0:
        labels = labels.groupby([LABEL_COL, SEASON_COL], group_keys=False).apply(
            lambda x: x.sample(frac=fraction)
        )

    return labels


def ts_transforms(
    ts: NDArray,
    dates: pd.Series,
    season: int,
    temperatures: Optional[NDArray] = None,
    start_month: int = 11,
    max_n_positions: int = 397,
    standardize: bool = False,
    augment: bool = False,
) -> Tuple[
    NDArray[np.float32], NDArray[np.int32], NDArray[np.int32], NDArray[np.uint8]
]:
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
        ts = ts + np.clip(np.random.normal(0, sigma, size=ts.shape), -1 * clip, clip)

    # Days normalizatioin to ref date
    days = [
        (date.fromisoformat(d) - date(year=season - 1, month=start_month, day=1)).days
        for d in dates
    ]

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
        ts_padded.astype(np.float32),
        positions_padded.astype(np.int32),
        days_padded.astype(np.int16),
        mask.astype(np.uint8),
    )
