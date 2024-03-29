import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import seaborn as sns


def patch_attention(m):
    """Patch attention in nn.TransformerEncoder
    to return averaged attention maps

    Args :
        m (nn.TransformerEncoder) : model

    """
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveAttentionMapHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


def merge(attn_maps_months):
    """Merge Attention Maps by averaging per month

    Args:
        attn_maps_months (pd.DataFrame) : Attention maps

    Returns:
        attn_maps_months avergaed
    """
    return (
        pd.concat(attn_maps_months)
        .groupby(["month", "layer", "target"], as_index=False)
        .mean()
    )


def resample(
    attention_map: NDArray,
    days: NDArray,
    masks: NDArray,
    targets: NDArray,
    ref_month: int = 11,
    ref_year: int = 2023,
):
    """Map attention maps from days to month to get a
    behavior on a monthly manner.


    Args:
        attention_map (NDArray) : Attention maps
        days (NDArray) : Time serie days
        masks (NDArray) : days masking
        targets (NDArray) : class
        ref_month (int = 11) : reference month for season
        ref_year (int = 2023) : reference year for season

    Returns:
        Merged attention maps on a average value per month on the
        season.
    """
    attn_maps_months = []
    for attn_maps, d, m, t in zip(attention_map, days, masks, targets):
        attn_maps = attn_maps[:, : m.sum(), : m.sum()].mean(axis=1)
        data = dict(enumerate(attn_maps))
        data["days"] = d[: m.sum()]
        df = pd.DataFrame(data).melt(
            id_vars="days",
            var_name="layer",
            value_name="map_value",
        )

        df["date"] = pd.to_datetime(
            pd.to_datetime(f"{ref_year - 1}-{ref_month}-01")
            + pd.to_timedelta(df["days"], unit="D")
        )
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df.loc[(df["year"] == ref_year - 1) & (df["month"] >= ref_month), "month"] = (
            df.loc[(df["year"] == ref_year - 1) & (df["month"] >= ref_month), "month"]
            - 12
        )
        if df["month"].min() <= 0:
            df["month"] += abs(df["month"].min())

        df_month = (
            df[["month", "layer", "map_value"]]
            .groupby(["month", "layer"], as_index=False)
            .sum()
        )

        df_month["target"] = t

        attn_maps_months.append(df_month)

    return merge(attn_maps_months)


def plot_attention(
    attn_maps_df: pd.DataFrame,
    step_name: str,
    post_title: str = "",
):
    """Plot attention maps

    Args:
        attn_maps_df (pd.DataFrame) : Attention maps
        step_name (str) : Name of the time step (e.g. month)
        post_title (str = "") : Post title (e.g. class name)
    """
    fig, ax = plt.subplots()

    for layer in attn_maps_df["layer"].unique():
        # Filter DataFrame for the current layer
        layer_df = attn_maps_df[attn_maps_df["layer"] == layer]
        steps = layer_df[step_name].values
        values = layer_df["map_value"].values

        values_norm = np.zeros(14)
        values_norm[steps] = values

        # Plot line
        g = sns.lineplot(
            x=range(14),
            y=values_norm,
            label=layer,
            ax=ax,
        )

        # Fill the area under the line
        plt.fill_between(range(14), 0, values_norm, alpha=0.2)

    g.set(title=f"Average Attention Map {post_title}", ylim=(0, 1))
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Layers")
    plt.tight_layout()

    return fig
