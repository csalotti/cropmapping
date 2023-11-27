from datetime import date, timedelta

import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray


def ndvi(red: NDArray, nir: NDArray):
    return np.where(nir != 0, (nir - red) / (nir + red), 0)


def plot_ndvi(
    days: NDArray,
    days_mask: NDArray,
    removed_days_mask,
    pred: NDArray,
    gd: NDArray,
    nir_index: int = 6,
    red_index: int = 2,
    start_year: int = 2022,
    start_month: int = 11,
    start_day: int = 1,
):
    days = days[days_mask]
    gd = gd[days_mask]
    pred = pred[days_mask]
    removed_days_mask = removed_days_mask[days_mask]

    dates = np.asarray([
        str(date(start_year, start_month, start_day) + timedelta(days=d))
        for d in days.tolist()
    ], dtype='datetime64[s]')

    ndvi_gd = ndvi(gd[:, red_index], gd[:, nir_index])
    ndvi_mask = ndvi(gd[:, red_index], gd[:, nir_index])
    ndvi_pred = ndvi(pred[:, red_index], pred[:, nir_index])
    ndvi_mask[removed_days_mask] = 0
    
    data = pd.DataFrame(
        {
            "days": dates,
            "gt": ndvi_gd,
            "gt_masked": ndvi_mask,
            "pred": ndvi_pred,
        }
    ).melt(
        id_vars="days",
        var_name="source",
        value_name="ndvi",
    )

    g = sns.lineplot(
        data=data,
        x="days",
        y="ndvi",
        style="source",
        markers=True,
        dashes=True,
    )

    g.tick_params(axis="x", labelrotation=45)

    return g.get_figure()
