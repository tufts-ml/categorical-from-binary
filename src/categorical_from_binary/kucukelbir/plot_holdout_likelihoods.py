import os
from typing import Optional

import seaborn as sns
from matplotlib import pyplot as plt


sns.set(style="whitegrid")

from pandas.core.frame import DataFrame

from categorical_from_binary.io import ensure_dir


def plot_performance_over_time_ADVI_vs_IB_CAVI(
    df_performance_advi: DataFrame,
    df_performance_cavi: DataFrame,
    save_dir: Optional[str] = None,
    experiment_description_for_filename_postfix: Optional[str] = None,
) -> None:
    """
    Arguments:
        df_performance_advi: One return value of the ADVI optimizer. A pandas Dataframe showing
            holdout performance metrics over time.
        df_performance_cavi: One return value of the IB_CAVI optimizer. A pandas Dataframe showing
            holdout performance metrics over time.
    """

    # aliases
    perf_advi = df_performance_advi
    perf_cavi = df_performance_cavi

    # time
    secs_advi = perf_advi["seconds elapsed"].to_numpy()
    secs_cavi = perf_cavi["seconds elapsed (cavi)"].to_numpy()

    # function:
    for metric_as_string in [
        "mean holdout log likelihood",
        "mean holdout likelihood",
        "correct classification rate",
    ]:

        metric_advi = perf_advi[f"{metric_as_string}"].to_numpy()
        if metric_as_string == "correct classification rate":
            metric_cavi = perf_cavi[f"{metric_as_string}"].to_numpy()
        else:
            metric_cavi = perf_cavi[f"{metric_as_string} for CBC_PROBIT"].to_numpy()
        plt.plot(secs_cavi, metric_cavi, linewidth=4)
        plt.plot(secs_advi, metric_advi, linewidth=4)
        plt.legend(["IB-CAVI", "ADVI"], fontsize=16)
        plt.xlabel("Time (secs)", fontsize=16)
        plt.ylabel(f"{metric_as_string}", fontsize=16)
        if save_dir is not None:
            ensure_dir(save_dir)
            metric_as_string_no_spaces = metric_as_string.replace(" ", "_")
            basename = f"{metric_as_string_no_spaces}_{experiment_description_for_filename_postfix}.png"
            filepath = os.path.join(save_dir, basename)
            plt.savefig(filepath, bbox_inches="tight")
        else:
            plt.show()
        plt.clf()
