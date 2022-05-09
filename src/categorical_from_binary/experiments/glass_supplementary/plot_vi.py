import numpy as np
import pandas as pd
import seaborn as sns

from categorical_from_binary.experiments.glass_supplementary.analysis import HoldoutLogLikes


# sns.set_theme(style="whitegrid")
sns.set(style="darkgrid")
import matplotlib.pyplot as plt


def make_plot_comparing_holdout_log_likes_for_ib_plus_sdo_vs_ib_plus_do(
    holdout_loglikes: HoldoutLogLikes,
):

    n_splits = len(holdout_loglikes.ib_plus_sdo)
    geom_means_sdo = np.exp(holdout_loglikes.ib_plus_sdo)
    geom_means_do = np.exp(holdout_loglikes.ib_plus_do)
    diffs_in_geom_means = geom_means_sdo - geom_means_do

    geom_means_sdo_sorted = [geom_means_sdo[i] for i in np.argsort(diffs_in_geom_means)]
    geom_means_do_sorted = [geom_means_do[i] for i in np.argsort(diffs_in_geom_means)]

    geom_means = geom_means_do_sorted + geom_means_sdo_sorted
    model_type = ["IB+CBC"] * n_splits + ["IB+CBM"] * n_splits
    split_ids = [i + 1 for i in range(n_splits)] * 2

    df = pd.DataFrame(
        {
            "split id": pd.Series(split_ids),
            "holdout log-likelihood (geometric mean)": pd.Series(geom_means),
            "model_type": pd.Series(model_type),
        }
    )

    sns.barplot(
        x="split id",
        y="holdout log-likelihood (geometric mean)",
        hue="model_type",
        data=df,
    )
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    # we can supply a rectangle box that the whole subplots area (including legend) will fit into
    # plt.tight_layout(rect=[0, 0, 0.8, 0.95])
    # plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.title("Mean probability across occurrences of the RAREST overall category")
    plt.legend(loc="upper right", frameon=False)
    plt.show()


def make_violin_plot_of_differences_in_geometric_mean_log_likes_for_ib_plus_sdo_vs_ib_plus_do(
    holdout_loglikes: HoldoutLogLikes,
):

    geom_means_sdo = np.exp(holdout_loglikes.ib_plus_sdo)
    geom_means_do = np.exp(holdout_loglikes.ib_plus_do)
    diffs_in_geom_means = geom_means_sdo - geom_means_do

    df = pd.DataFrame(
        {
            "differences in geometric means": pd.Series(diffs_in_geom_means),
        }
    )

    sns.violinplot(x="differences in geometric means", data=df)
    plt.show()
