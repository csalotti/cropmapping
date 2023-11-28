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


def plot_attention(attention_map, days, mask):
    attention_map = attention_map[:, : mask.sum(), : mask.sum()].mean(-2)
    days = days[: mask.sum()]

    data = dict(enumerate(attention_map))
    data["days"] = days

    df = pd.DataFrame(data).melt(
        id_vars="days",
        var_name="layer",
        value_name="map_value",
    )
    for layer in df["layer"].unique():
        # Filter DataFrame for the current layer
        layer_df = df[df["layer"] == layer]

        # Plot line
        g = sns.lineplot(
            x="days", y="map_value", data=layer_df, label=layer, marker="o"
        )

        # Fill the area under the line
        plt.fill_between(layer_df["days"], 0, layer_df["map_value"], alpha=0.2)

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Layers")
    plt.tight_layout()

    return g.get_figure()
