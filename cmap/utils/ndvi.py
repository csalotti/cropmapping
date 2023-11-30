from datetime import date, timedelta

import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

import matplotlib.pyplot as plt

def ndvi(red: NDArray, nir: NDArray):
    return np.where(nir != 0, (nir - red) / (nir + red), 0)


def plot_ndvi(
    days: NDArray,
    days_mask: NDArray,
    removed_days_mask,
    pred: NDArray,
    gt: NDArray,
    nir_index: int = 6,
    red_index: int = 2,
    start_year: int = 2022,
    start_month: int = 11,
    start_day: int = 1,
):
    days_mask = days_mask == 1
    days = days[days_mask]
    gt = gt[days_mask]
    pred = pred[days_mask]
    removed_days_mask = removed_days_mask[days_mask].astype('bool')

    dates = np.asarray(
        [
            str(date(start_year, start_month, start_day) + timedelta(days=d))
            for d in days.tolist()
        ],
        dtype="datetime64[s]",
    )

    ndvi_gt = ndvi(gt[:, red_index], gt[:, nir_index])
    ndvi_pred = ndvi(pred[:, red_index], pred[:, nir_index])
    ndvi_pred[~removed_days_mask] = ndvi_gt[~removed_days_mask]

    data = pd.DataFrame(
        {
            "days": dates,
            "gt": ndvi_gt,
            "pred": ndvi_pred,
        }
    ).melt(
        id_vars="days",
        var_name="source",
        value_name="ndvi",
    )

    fig, ax = plt.subplots()

    g = sns.lineplot(
        data=data,
        x="days",
        y="ndvi",
        hue="source",
        style="source",
        markers=True,
        dashes=True,
        ax=ax
    )

    g.tick_params(axis="x", labelrotation=45)

    return fig
