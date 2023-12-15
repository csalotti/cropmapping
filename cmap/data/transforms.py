from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cmap.utils.constants import LABEL_COL, SEASON_COL


def labels_sample(
    labels: pd.DataFrame,
    fraction: float,
    seasons: List[int],
    classes: List[str],
    subsample: bool = False,
) -> pd.DataFrame:
    """Labels sampling

    Filter labels per season, sub sample other class to match
    second highest label and reduce number of sample
    with respect to class distribution

    Args:
        labels (pd.DataFrame): Labels DataFrame
        fraction (float):  Sampling labels fraction
        seasons (List[int]): Seasons to keep
        classes (List[str]) : classes to consider

    """
    # Season and classes filtering
    labels = labels.query(f"{SEASON_COL} in @seasons").query(f"{LABEL_COL} in @classes")

    # Subsampling 'other' class that should be equal to the maximum
    # positive (non-other) class for each seson
    if subsample:
        points_other = labels.query(f"{LABEL_COL} == 'other'")
        points_positives = labels.query(f"{LABEL_COL} != 'other'")
        top_seasons_dist = (
            points_positives[[LABEL_COL, SEASON_COL]]
            .value_counts()
            .groupby(SEASON_COL)
            .max()
            .to_dict()
        )
        points_other = points_other.groupby(
            [LABEL_COL, SEASON_COL], group_keys=False
        ).apply(lambda x: x.sample(top_seasons_dist[x.name[1]]))
        labels = pd.concat([points_other, points_positives], ignore_index=True)

    # Subsample dataset respecting distribution of classes
    if fraction < 1.0:
        labels = labels.groupby([LABEL_COL, SEASON_COL], group_keys=False).apply(
            lambda x: x.sample(frac=fraction)
        )

    return labels


def ts_transforms(
    ts: NDArray,
    dates: NDArray,
    season: int,
    temperatures: Optional[NDArray] = None,
    start_month: int = 11,
    max_n_positions: int = 80,
    standardize: bool = False,
    augment: bool = False,
) -> Tuple[
    NDArray[np.float32], NDArray[np.int32], NDArray[np.int32], NDArray[np.uint8]
]:
    """Features transformations on time seris features.
    All features are normalized and pos-padded to fit max_n_positions.
    Bands are divided by 10_000 (startdizeed = False) or standadized.
    Dates are transformed into days from a starting sason day.
    Tempratures are used as positions if provided, otherwise, days are used.
    If augmentation is enabled, a nois N(0,1e-2) cliped at 3e-2 is applied
    to each bands

    Args:
        ts (NDArray) : Bands data
        dates (NDArray) : time steps
        season (int) :  Current season
        temperatures (Optional[NDArray]) : Corresponding temperatures,
        start_month (int) : Season start month (season -1)
        max_n_positions (int) : maximum number of positions
        standardize (bool) : standardization flag ,
        augment (bool) : Augmentation flag

    Returns:
        (ts_norm_paddeed, days_norm_padded, positions_norm_padded, mask_padded)
    """
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
    days = [(d - date(year=season - 1, month=start_month, day=1)).days for d in dates]

    # GDD computation
    if temperatures is not None:
        temperatures = np.cumsum(temperatures)
        temperatures = temperatures[days]
    
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
