import pickle

import numpy as np
import pandas as pd
import seaborn as sns


sns.set(style="darkgrid")
from typing import List, Optional

import matplotlib.pyplot as plt

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.evaluate.multiclass import (
    BetaType,
    DataType,
    Measurement,
    Metric,
    form_dict_mapping_measurement_context_to_values,
)


def context_is_relevant(context, data_generating_link_of_interest):
    return (
        context.data_type == DataType.TEST
        and context.metric == Metric.MEAN_LOG_LIKELIHOOD
        and context.link_for_generating_data == data_generating_link_of_interest
    )


def make_relevant_subdict(d, data_generating_link_of_interest):
    subdict = {}
    for context, values in d.items():
        if context_is_relevant(context, data_generating_link_of_interest):
            subdict[context] = values
    return subdict


def make_figure_comparing_CBM_and_CBC_across_data_generating_links(
    measurements: List[Measurement],
    data_generating_links_of_interest: Optional[List[Link]] = None,
):
    """
    Make a figure comparing IB+CBM vs IB+CBC across various data generating links.
    The figure is a seaborn boxplot (https://seaborn.pydata.org/generated/seaborn.boxplot.html).
    Along the way, we create a pandas dataframe showing reduction of relative error that could be
    split off into its own function.

    Arguments:
        data_generating_links_of_interest: By default, this is inferred from the data generating
        links that exist in the measurements, but it can be specified if one wants the figure to
        present them in a certain order (or if one wants to plot a mere subset of what's available
        in the measurements).
    Returns:
        Instance of matplotlib.axes._subplots.AxesSubplot
    """

    d = form_dict_mapping_measurement_context_to_values(measurements)

    data_generating_links = []
    error_reductions = []

    if data_generating_links_of_interest is None:
        data_generating_links_of_interest = list(
            set([m.context.link_for_generating_data for m in measurements])
        )

    for data_generating_link_of_interest in data_generating_links_of_interest:
        sd = make_relevant_subdict(d, data_generating_link_of_interest)
        for context, values in sd.items():
            if (
                context.link_for_category_probabilities == Link.CBC_PROBIT
                and context.beta_type == BetaType.VARIATIONAL_POSTERIOR_MEAN
            ):
                values_with_CBC_probit_estimator = values
            elif (
                context.link_for_category_probabilities == Link.CBM_PROBIT
                and context.beta_type == BetaType.VARIATIONAL_POSTERIOR_MEAN
            ):
                values_with_CBM_probit_estimator = values
            elif (
                context.link_for_category_probabilities
                == data_generating_link_of_interest
                and context.beta_type == BetaType.GROUND_TRUTH
            ):
                values_ground_truth = values
            else:
                raise ValueError("Something weird happened")

        f = lambda x: np.array(x) * 1  # f=np.exp
        CBC = f(values_with_CBC_probit_estimator)
        CBM = f(values_with_CBM_probit_estimator)
        GT = f(values_ground_truth)
        print(
            f"For data generating link {data_generating_link_of_interest} "
            f"the mean held out log likelihood for CBM was {np.mean(CBM):.02} and "
            f"with the ground truth was  {np.mean(GT):.02} "
        )

        CBC_dists = GT - CBC
        CBM_dists = GT - CBM
        error_reductions_curr = (CBC_dists - CBM_dists) / CBC_dists
        data_generating_links_curr = [data_generating_link_of_interest.name] * len(
            error_reductions_curr
        )
        # print(f"{error_reductions_curr}")
        error_reductions.extend(error_reductions_curr)
        data_generating_links.extend(data_generating_links_curr)

    df = pd.DataFrame(
        {
            "relative error reduction": pd.Series(error_reductions),
            "data generating link": pd.Series(data_generating_links),
        }
    )

    # TODO: split above off into its own function. The construction of the pandas
    # dataframe should be its own function

    ###
    # This part makes the figure
    ###

    # boxplot
    # https://seaborn.pydata.org/generated/seaborn.boxplot.html
    plt.figure()
    ax = sns.boxplot(x="data generating link", y="relative error reduction", data=df)
    ax = sns.swarmplot(
        x="data generating link", y="relative error reduction", data=df, color=".25"
    )
    return ax


###
# Main Part of Module
###

RECREATE_UAI_TPM_FIGURE = False

if RECREATE_UAI_TPM_FIGURE:
    with open("data/measurements_on_CBM_vs_CBC_estimators.pkl", "rb") as handle:
        measurements = pickle.load(handle)

ax = make_figure_comparing_CBM_and_CBC_across_data_generating_links(
    measurements,
    data_generating_links_of_interest=[
        Link.CBC_PROBIT,
        Link.CBM_PROBIT,
        Link.MULTI_PROBIT,
        Link.MULTI_LOGIT,
    ],
)
# plt.show()
# plt.savefig(NAME_ME)
# plt.close()
