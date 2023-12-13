from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def get_dist_plot(labels: List[str]):
    labels_dist = pd.DataFrame(labels, columns=["label"]).value_counts()
    # Create a bar plot with adjusted bar width
    ax = labels_dist.plot.bar(
        x="label", y="count", color="skyblue", edgecolor="black", width=0.8
    )

    # Set labels and title
    ax.set_xlabel("Label", fontsize=12)
    ax.set_title("Distribution of Labels", fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Add data labels on top of the bars
    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    # Remove frame and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])

    # Remove y-axis title
    ax.set_ylabel("")

    # Remove legend if it exists
    ax.legend().set_visible(False)

    # Display the plot
    plt.tight_layout()

    # Return the generated figure
    return plt.gcf()
