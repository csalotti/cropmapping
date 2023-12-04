from datetime import date
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def patch_attention(m):
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


def resample(attention_map, days, masks, classes, ref_month=11, ref_year=2023):
    attn_maps_month = []
    for attn_maps, d, m, c in zip(attention_map, days, masks, classes):
        attn_maps = attn_maps[:, : m.sum(), : m.sum()]
        d = d[: m.sum()]
        data = dict(enumerate(attn_maps))
        data["days"] = d
        df = pd.DataFrame(data).melt(
            id_vars="days",
            var_name="layer",
            value_name="map_value",
        )

        df["date"] = pd.to_datetime(
            date(year=ref_year - 1, month=ref_month, day=1) + df["days"]
        )
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df.query("(year == {ref_year}) & (month >= ref_month)")["month"] = (
            df.query("(year == {ref_year}) & (month >= ref_month)")["month"] + 1
        )
        df_month = (
            df[["month", "layer", "map_value"]]
            .groupby(["month", "layer"], as_index=False)
            .sum()
        )

        df_month["class"] = c

        attn_maps_month.append(df_month)

    return (
        pd.concat(attn_maps_month)
        .groupby(["month", "layer", "class"], as_index=False)
        .mean()
    )


def plot_attention(attn_maps_df, step_name: str, post_title: str = ""):
    fig, ax = plt.subplots()

    for layer in attn_maps_df["layer"].unique():
        # Filter DataFrame for the current layer
        layer_df = attn_maps_df[attn_maps_df["layer"] == layer]

        # Plot line
        g = sns.lineplot(
            x=step_name,
            y="map_value",
            data=layer_df,
            label=layer,
            ax=ax,
        )

        # Fill the area under the line
        plt.fill_between(layer_df[step_name], 0, layer_df["map_value"], alpha=0.2)

    g.set(
        title=f"Average Attention map {post_title}",
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Layers")
    plt.tight_layout()

    return fig
