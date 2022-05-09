"""
To create mock data for this, do

python src/categorical_from_binary/experiments/sims/run_sims.py  --seed 0 --test_run true  --list_of_scales_for_predictive_categories  0.1 2.0
python src/categorical_from_binary/experiments/sims/run_sims.py  --seed 1 --test_run true  --list_of_scales_for_predictive_categories  0.1 2.0

This module should be run locally after scp'ing the data from `run_sims` to a local results_dir
"""


import numpy as np


np.set_printoptions(precision=3, suppress=True)

import seaborn as sns


sns.set(style="darkgrid")
from matplotlib import pyplot as plt

from categorical_from_binary.experiments.compare_to_mle.results_collector import (
    make_df_from_directory_taking_one_file_for_each_seed,
)


# results_dir="./tmp"
# need to scp -r mwojno01@login.cluster.tufts.edu:/cluster/tufts/hugheslab/mwojno01/data/results/sims/
# from results_dir_cluster to results_dir_local
results_dir_cluster = "/cluster/tufts/hugheslab/mwojno01/data/results/sims/"
results_dir_local = "/Users/mwojno01/Repos/categorical_from_binary/data/results/sims_results_in_icml_draft/ib_probit/"
df = make_df_from_directory_taking_one_file_for_each_seed(
    results_dir_local, prefix_for_filenames_of_interest="sim_results_CAVI_vs_MLE_"
)
# TODO : time for IB CAVI increases with dataset sizes (up to 30 min!), whereas MLE does not do this...(never exeeds 1 min).
# Should we be concerned? Guess its not a big deal since MLE's problem is with smaller datasets, not bigger ones.  Still not ideal, though.

###
# now try to add in plotting stuff from the old module
###
x_label = "Mean covariate-conditional entropy"  # "Mean covariate-conditional category entropy"
y_label = "Mean holdout log likelihood"
outcome_of_interest_postfix = "log_like"
dir_for_saving_figures = "tmp/"
new_legend_title = "Prediction Method"
# Warning! The new labels need to be in the same order as the plot's default legend.  TODO: Fix this.


fig, axes = plt.subplots(
    1,
    2,
    figsize=(5, 3),
    sharey=True,
    sharex=True,
    squeeze=True,
    constrained_layout=True,
)
Ks = list(set(df["K"]))
Ns = list(set(df["N"]))
Ms = list(set(df["M"]))
K = Ks[0]
M = Ms[0]

legend = True
for (n, N) in enumerate(reversed(Ns)):
    ## select context of interest for this subplot
    dff = df.query(f"K == {K} & N == {N} & M == {M}")

    ## compute the mean covariate conditional entropy over seeds
    dff["mean_covariate_conditional_entropy_over_seeds"] = ""
    for scale in set(dff.scale_for_predictive_categories):
        mean_covariate_conditional_entropy_over_seeds = dff[
            dff.scale_for_predictive_categories == scale
        ].mean_covariate_conditional_entropy_of_true_category_probabilities.mean()
        dff.loc[
            dff.scale_for_predictive_categories == scale,
            "mean_covariate_conditional_entropy_over_seeds",
        ] = mean_covariate_conditional_entropy_over_seeds

    # mostly doing this so x axis isn't overly precise in the plot.
    dff = dff.round(decimals=3)

    dff = dff.loc[
        :,
        dff.columns.str.endswith(outcome_of_interest_postfix)
        | dff.columns.str.startswith("mean_covariate_conditional_entropy_over_seeds"),
    ]

    # 'melting' here basically takes a bunch of different columns and makes them different factors
    # for a single column.  So the dataframe gets longer and thinner.
    df_melted = dff.melt(
        "mean_covariate_conditional_entropy_over_seeds",
        var_name="method",
        value_name=outcome_of_interest_postfix,
    )

    g = sns.lineplot(
        x="mean_covariate_conditional_entropy_over_seeds",
        y=outcome_of_interest_postfix,
        hue="method",
        data=df_melted,
        legend=True,
        estimator="mean",
        ax=axes[n],
    )
    handles, labels = axes[n].get_legend_handles_labels()
    axes[n].legend().set_visible(False)
    axes[n].set(xlabel=None)
    axes[n].set(ylabel=None)


fig.supylabel(y_label)
fig.supxlabel(x_label)

# TODO: These should NOT be hardcoded!
cols = ["N=1P", "N=100P"]

pad = 6  # in points

for ax, col in zip(axes, cols):
    ax.annotate(
        col,
        xy=(0.5, 1),
        xytext=(0, pad),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="center",
        va="baseline",
        fontsize=14,
    )


# ### Get labels from last axes and make legend
# handles, labels = axes[0].get_legend_handles_labels()
# h,l = ax.get_legend_handles_labels()

new_legend_labels = [
    "Generating Process",
    "Baserate Frequency",
    "Softmax (via MLE)",
    "CB-Probit (via IB-CAVI)",
]
new_labels = new_legend_labels


fig.legend(
    [handles[0], handles[3], handles[2], handles[1]],
    [new_labels[0], new_labels[3], new_labels[2], new_labels[1]],
    loc="upper left",
    bbox_to_anchor=(0.57, 0.405, 0.5, 0.1),
    fontsize="small",
)
plt.show()
