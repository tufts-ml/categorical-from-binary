"""
Why?

1. Can get MLE of CBM and MLE of IB to see if they line up as n->infty 
2. Want to get variance in a category prob as a fucntion of the variance in beta.

For now let's just use binary CBM.
Reference:  https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf

"""
import os

import numpy as np


np.set_printoptions(precision=3, suppress=True)

import typing
from typing import List, Optional

import seaborn as sns
from pandas.core.frame import DataFrame


sns.set(style="darkgrid")
from matplotlib import pyplot as plt

from categorical_from_binary.io import ensure_dir


###
# Plotting functions
###
# Plotting functions are located in the same module because they will only work if the names
# match those assigned in the function creating the DataFrame


def _alter_font_size_and_title_and_labels_for_legend_of_facetgrid(
    g: sns.axisgrid.FacetGrid,
    size: int,
    new_title: typing.Optional[str] = None,
    new_labels: typing.Optional[List[str]] = None,
) -> sns.axisgrid.FacetGrid:
    # https://stackoverflow.com/questions/45201514/how-to-edit-a-seaborn-legend-title-and-labels-for-figure-level-functions
    legend = g.get_legend()
    if new_title is not None:
        legend.set_title(new_title)
    if new_labels is not None:
        for t, l in zip(legend.texts, new_labels):
            t.set_text(l)
    # change font size for legend entries
    plt.setp(legend.get_texts(), fontsize=size)
    # change font size for legend title
    plt.setp(legend.get_title(), fontsize=size)
    return g


def make_plot_of_performance_results_for_one_context(
    df: DataFrame,
    N: int,
    M: int,
    K: int,
    outcome_of_interest_postfix: str,
    x_label: str = "",
    y_label: str = "",
    font_size_title: int = 12,
    font_size_axis_labels: int = 12,
    font_size_legend: int = 12,
    legend: bool = True,
    new_legend_title: typing.Optional[str] = None,
    new_legend_labels: typing.Optional[List[str]] = None,
    minimum_scale_for_predictive_categories: Optional[float] = None,
) -> sns.axisgrid.FacetGrid:
    """
    A "context" is a combination of N, M, and K, defined below.

    Arguments:
        df: Output of `run_performance_simulations_on_softmax_data`.
        N: number of samples
        M: number of covariates
        K: number of categories
        input_of_interest_prefix: should be the beginning of a column name; this will become the x-axis in the plot
        outcome_of_interest_postfix: should be the end of a column name; this will become the y-axis in the plot
    """
    ## select context of interest for this subplot
    dff = df.query(f"K == {K} & N == {N} & M == {M}")

    if minimum_scale_for_predictive_categories is not None:
        dff = dff.query(
            f"scale_for_predictive_categories>={minimum_scale_for_predictive_categories}"
        )
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

    sns.set(rc={"figure.figsize": (5, 8)})
    g = sns.lineplot(
        x="mean_covariate_conditional_entropy_over_seeds",
        y=outcome_of_interest_postfix,
        hue="method",
        data=df_melted,
        legend=legend,
        estimator="mean",
    )
    # title = f"K={int(K)}, M={int(M)}, N={int(N/(K*M))}P"
    # title = f"{K} categories, {M} covariates, {N} samples"
    title = f"N={int(N/(K*(M+1)))}P"
    g.set_title(title, size=font_size_title)
    g.set_ylabel(y_label, size=font_size_axis_labels)
    g.set_xlabel(x_label, size=font_size_axis_labels)
    if new_legend_title is not None or new_legend_labels is not None:
        g = _alter_font_size_and_title_and_labels_for_legend_of_facetgrid(
            g,
            font_size_legend,
            new_legend_title,
            new_legend_labels,
        )
    return g


def plot_performance_results_for_many_data_contexts_in_succession(
    df: DataFrame,
    outcome_of_interest_postfix: str,
    x_label: str = "",
    y_label: str = "",
    font_size_title: int = 12,
    font_size_axis_labels: int = 12,
    font_size_legend: int = 12,
    new_legend_title: typing.Optional[str] = None,
    new_legend_labels: typing.Optional[List[str]] = None,
    minimum_scale_for_predictive_categories: Optional[float] = None,
    dir_for_saving_figures: Optional[str] = None,
    show_plots: bool = True,
) -> None:
    """
    Note: we first iteratively display all the plots without the legend, since this is more useful
    if we will be putting these into a Latex table of figures.  Then we display the last plot one
    extra time but with the legend, so we can screenshot it.
    """
    Ks = list(set(df["K"]))
    Ns = list(set(df["N"]))
    Ms = list(set(df["M"]))
    for K in Ks:
        for M in Ms:
            for N in Ns:
                print(f"Now making plot for data context N={N}, K={K}, M={M}.")
                make_plot_of_performance_results_for_one_context(
                    df,
                    N,
                    M,
                    K,
                    outcome_of_interest_postfix,
                    x_label,
                    y_label,
                    font_size_title,
                    font_size_axis_labels,
                    font_size_legend,
                    legend=True,
                    new_legend_title=new_legend_title,
                    new_legend_labels=new_legend_labels,
                    minimum_scale_for_predictive_categories=minimum_scale_for_predictive_categories,
                )
                plt.tight_layout(rect=(0, 0, 1, 0.975))
                if dir_for_saving_figures is not None:
                    child_directory = os.path.join(
                        dir_for_saving_figures, outcome_of_interest_postfix
                    )
                    ensure_dir(child_directory)
                    filename = os.path.join(child_directory, f"K={K}_N={N}")
                    plt.savefig(filename)
                if show_plots:
                    plt.show()
                plt.close("all")
