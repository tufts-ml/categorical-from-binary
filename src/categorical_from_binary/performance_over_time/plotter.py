import os
import warnings
from typing import Dict, Optional, Union

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


sns.set(style="whitegrid")

from pandas.core.frame import DataFrame

from categorical_from_binary.io import ensure_dir
from categorical_from_binary.performance_over_time.classes import (
    PerformanceOverTimeResults,
)


# TODO: Don't REQUIRE IB-CAVI and ADVI to be here... make everything optional
# except the DGP


def plot_performance_over_time(
    df_performance_advi_by_lr: Dict[float, DataFrame],
    df_performance_cavi_probit: DataFrame,
    df_performance_cavi_logit: Optional[DataFrame] = None,
    df_performance_nuts: Optional[DataFrame] = None,
    df_performance_softmax_via_pga_and_gibbs: Optional[DataFrame] = None,
    save_dir: Optional[str] = None,
    log_like_data_generating_process: Optional[float] = None,
    accuracy_data_generating_process: Optional[float] = None,
    log_like_random_guessing: Optional[float] = None,
    accuracy_random_guessing: Optional[float] = None,
    min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve=0.0,
    min_log_likelihood_for_y_axis: Optional[Union[str, float]] = None,
    max_log_likelihood_for_y_axis: Optional[float] = None,
    add_legend_to_plot: bool = True,
    show_cb_logit: bool = True,
    label_advi_lrs_by_index: bool = False,
    save_legend_separately: bool = False,
    CBC_name: str = "CBC",
    CBM_name: str = "CBM",
) -> None:
    """
    Arguments:
         df_performance_advi_by_lr: Dict mapping learning rate to a pandas Dataframe showing
            holdout performance metrics over time.
        df_performance_cavi_probit: One return value of the IB_CAVI optimizer. A pandas Dataframe showing
            holdout performance metrics over time.
        min_log_likelihood_for_y_axis : float or "random guessing"
            if  "random guessing" and log_like_random_guessing /  accuracy_random_guessing is not None
            we use random guessing as the bottom of the y-axis
        min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve: Optional float.
            Moving this up from the default of 0.0
            can prevent situations where we give legend entries to curves that are not there.
        CBC_name: Can be set to "DO" for backwards compatability
        CBM_name: Can be set to "SDO" for backwards compatability
    """
    plt.clf()

    EPSILON_TO_AVOID_EXACT_ZERO_SECONDS_WHEN_LOGGING = (
        df_performance_cavi_probit["seconds elapsed"][1] / 2
    )

    warnings.warn(
        f"We log time on the x-axis. But to do this, "
        f"we do some surgery on the initial timestamp from each method, replacing it with some epsilon."
        f"This could wreak havoc if the initial performance measurement is NOT taken at time zero, as assumed."
    )

    # TODO: Construct an enum so we give consistent names to the various methods.

    # construct color palettes
    # Reference: https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=6
    colors_for_advi = sns.color_palette(
        palette="BuPu", n_colors=len(df_performance_advi_by_lr)
    )
    colors_for_cavi = sns.color_palette(palette="YlOrRd", n_colors=2)
    colors_for_other_methods = sns.color_palette(palette="BuGn", n_colors=6)

    list_of_metrics_as_strings = [
        "test mean log likelihood",
        "test mean likelihood",
        "test accuracy",
        "train mean log likelihood",
        "train mean likelihood",
        "train accuracy",
    ]
    for (m, metric_as_string) in enumerate(list_of_metrics_as_strings):

        # Keep track of the best performing method (so we can set ylim)
        max_metric = -np.inf
        min_metric = np.inf

        ###
        #  CAVI-Probit
        ###

        # Pick link for CB-Probit using "cheap BMA"
        last_train_ll_CBC = df_performance_cavi_probit[
            f"train mean log likelihood with {CBC_name}_PROBIT"
        ].iloc[-1]
        last_train_ll_CBM = df_performance_cavi_probit[
            f"train mean log likelihood with {CBM_name}_PROBIT"
        ].iloc[-1]
        if last_train_ll_CBC > last_train_ll_CBM:
            cb_probit_link_as_string_with_cheap_BMA = f"{CBC_name}_PROBIT"
        else:
            cb_probit_link_as_string_with_cheap_BMA = f"{CBM_name}_PROBIT"

        # compute line to plot
        perf_cavi_probit = df_performance_cavi_probit
        secs_cavi_probit = perf_cavi_probit["seconds elapsed"].to_numpy()
        secs_cavi_probit[0] = EPSILON_TO_AVOID_EXACT_ZERO_SECONDS_WHEN_LOGGING
        metric_cavi_probit = perf_cavi_probit[
            f"{metric_as_string} with {cb_probit_link_as_string_with_cheap_BMA}"
        ].to_numpy()
        max_metric = np.nanmax([max_metric, np.nanmax(metric_cavi_probit)])
        min_metric = np.nanmin([min_metric, np.nanmin(metric_cavi_probit)])
        plt.plot(
            secs_cavi_probit,
            metric_cavi_probit,
            label="CB-Probit+IB-CAVI",
            linewidth=4,
            color=colors_for_cavi[-1],
        )

        ###
        #  CAVI-Logit
        ###
        # Optionally make a line for CAVI-Logit
        if df_performance_cavi_logit is not None and show_cb_logit is True:
            # Pick link for CB-Logit using "cheap BMA"
            last_train_ll_CBC = df_performance_cavi_logit[
                f"train mean log likelihood with {CBC_name}_LOGIT"
            ].iloc[-1]
            last_train_ll_CBM = df_performance_cavi_logit[
                f"train mean log likelihood with {CBM_name}_LOGIT"
            ].iloc[-1]
            if last_train_ll_CBC > last_train_ll_CBM:
                cb_logit_link_as_string_with_cheap_BMA = f"{CBC_name}_LOGIT"
            else:
                cb_logit_link_as_string_with_cheap_BMA = f"{CBM_name}_LOGIT"

            perf_cavi_logit = df_performance_cavi_logit
            secs_cavi_logit = perf_cavi_logit["seconds elapsed"].to_numpy()
            secs_cavi_logit[0] = EPSILON_TO_AVOID_EXACT_ZERO_SECONDS_WHEN_LOGGING
            metric_cavi_logit = perf_cavi_logit[
                f"{metric_as_string} with {cb_logit_link_as_string_with_cheap_BMA}"
            ].to_numpy()
            max_metric = np.nanmax([max_metric, np.nanmax(metric_cavi_logit)])
            min_metric = np.nanmin([min_metric, np.nanmin(metric_cavi_logit)])
            plt.plot(
                secs_cavi_logit,
                metric_cavi_logit,
                label="CB-Logit+IB-CAVI",
                linewidth=4,
                color=colors_for_cavi[-2],
            )

        ###
        # ADVI
        ###

        # make a line for ADVI for each learning rate
        for i, (lr, perf_advi) in enumerate(df_performance_advi_by_lr.items()):
            secs_advi = perf_advi["seconds elapsed"].to_numpy()
            secs_advi[0] = EPSILON_TO_AVOID_EXACT_ZERO_SECONDS_WHEN_LOGGING
            # TODO: Don't hardcode the link here
            metric_advi = perf_advi[f"{metric_as_string} with SOFTMAX"].to_numpy()
            max_metric = np.nanmax([max_metric, np.nanmax(metric_advi)])
            min_metric = np.nanmin([min_metric, np.nanmin(metric_advi)])
            total_n_its = len(metric_advi)
            iterate_to_check_for_nan = (
                int(
                    min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve
                    * total_n_its
                )
                - 1
            )
            if not np.isnan(metric_advi[iterate_to_check_for_nan]):
                if label_advi_lrs_by_index:
                    label_advi = r"Softmax+ADVI (lr=$\mathcal{L}$" + f"[{i}])"
                else:
                    label_advi = f"Softmax+ADVI (lr={lr})"
                # we reduce alpha (visibility) for higher LR's, since for our expts those suck
                plt.plot(
                    secs_advi,
                    metric_advi,
                    label=label_advi,
                    alpha=1.0 - (i + 1) / (len(df_performance_advi_by_lr) + 1),
                    linewidth=4,
                    color=colors_for_advi[i],
                )

        ###
        # NUTS
        ###
        # Optionally make a line for NUTS
        if df_performance_nuts is not None:
            perf_nuts = df_performance_nuts
            secs_nuts = perf_nuts["seconds elapsed"].to_numpy()
            # secs_nuts[0] = EPSILON_TO_AVOID_EXACT_ZERO_SECONDS_WHEN_LOGGING
            # TODO: don't hardcode the link here
            metric_nuts = perf_nuts[f"{metric_as_string} with SOFTMAX"].to_numpy()
            max_metric = np.nanmax([max_metric, np.nanmax(metric_nuts)])
            min_metric = np.nanmin([min_metric, np.nanmin(metric_nuts)])
            plt.plot(
                secs_nuts,
                metric_nuts,
                label="Softmax+NUTS",
                linewidth=4,
                color=colors_for_other_methods[-1],
                # alpha=1.0 - 1 / (len(colors_for_other_methods) + 1),
            )

        ###
        # Gibbs for Softmax with PGA
        ###
        # Optionally make a line for Gibbs softmax with PGA
        if df_performance_softmax_via_pga_and_gibbs is not None:
            perf_gibbs = df_performance_softmax_via_pga_and_gibbs
            secs_gibbs = perf_gibbs["seconds elapsed"].to_numpy()
            # secs_gibbs[0] = EPSILON_TO_AVOID_EXACT_ZERO_SECONDS_WHEN_LOGGING
            # TODO: don't hardcode the link here
            metric_gibbs = perf_gibbs[f"{metric_as_string} with SOFTMAX"].to_numpy()
            max_metric = np.nanmax([max_metric, np.nanmax(metric_gibbs)])
            min_metric = np.nanmin([min_metric, np.nanmin(metric_gibbs)])
            plt.plot(
                secs_gibbs,
                metric_gibbs,
                label="Softmax+PGA-Gibbs",
                linewidth=4,
                color=colors_for_other_methods[-2],
                alpha=1.0 - 2 / (len(colors_for_other_methods) + 1),
            )

        ###
        # Anchors
        ###

        # Optionally make a horizontal line for data generating process
        if (
            "log likelihood" in metric_as_string
            and log_like_data_generating_process is not None
        ):
            max_metric = np.nanmax([max_metric, log_like_data_generating_process])
            min_metric = np.nanmin([min_metric, log_like_data_generating_process])
            plt.axhline(
                y=log_like_data_generating_process,
                color="k",
                label="True Model (Softmax)",
                linestyle="--",
            )

        # Optionally make a horizontal line for random guessing
        if (
            "log likelihood" in metric_as_string
            and log_like_random_guessing is not None
        ):
            max_metric = np.nanmax([max_metric, log_like_random_guessing])
            min_metric = np.nanmin([min_metric, log_like_random_guessing])
            plt.axhline(
                y=log_like_random_guessing,
                color="k",
                label="Random guessing",
                linestyle="-",
            )

        # Optionally make a horizontal line for data generating process
        if (
            "accuracy" in metric_as_string
            and accuracy_data_generating_process is not None
        ):
            max_metric = np.nanmax([max_metric, accuracy_data_generating_process])
            min_metric = np.nanmin([min_metric, accuracy_data_generating_process])
            plt.axhline(
                y=accuracy_data_generating_process,
                color="k",
                label="True Model (Softmax)",
                linestyle="--",
            )

        # Optionally make a horizontal line for random guessing
        if "accuracy" in metric_as_string and accuracy_random_guessing is not None:
            max_metric = np.nanmax([max_metric, accuracy_random_guessing])
            min_metric = np.nanmin([min_metric, accuracy_random_guessing])
            plt.axhline(
                y=accuracy_random_guessing,
                color="k",
                label="Random guessing",
                linestyle="-",
            )

        ###
        # Set y and x limits
        ###

        if "log likelihood" in metric_as_string:
            if isinstance(min_log_likelihood_for_y_axis, float):
                plt.ylim(bottom=min_log_likelihood_for_y_axis)
            else:
                if min_log_likelihood_for_y_axis == "random guessing":
                    bottom_base = log_like_random_guessing
                else:
                    bottom_base = min_metric
                multiplier = 1.20 if bottom_base < 0.0 else 0.80
                plt.ylim(bottom=bottom_base * multiplier)
            if max_log_likelihood_for_y_axis is not None:
                plt.ylim(top=max_log_likelihood_for_y_axis)
            # elif log_like_data_generating_process is not None:
            #     # default value for y max, will be overriden
            #     # if max_log_likelihood_for_y_axis is not None
            #     plt.ylim(
            #         top=log_like_data_generating_process * dilation_factor_for_top_y
            #     )
            else:
                multiplier = 0.975 if max_metric < 0 else 1.025
                plt.ylim(top=max_metric * multiplier)

        plt.xlim(left=EPSILON_TO_AVOID_EXACT_ZERO_SECONDS_WHEN_LOGGING)

        # add labels and legend
        plt.xlabel("Time (secs)", fontsize=24)
        plt.xscale("log")
        plt.ylabel(f"{metric_as_string.capitalize()}", fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        if add_legend_to_plot:
            legend = plt.legend(
                shadow=True,
                fancybox=True,
                bbox_to_anchor=(1, 0.5, 0.3, 0.2),
                borderaxespad=0,
            )
        # plt.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
        # The above code means the legend box is positioned at the axes coordinate
        # (1.05, 1.0) that has the width of 0.3 and the height of 0.2,
        # where (1.05, 1.0) is the coordinate of the upper left corner of the legend bounding box.

        plt.tight_layout()

        # save or show
        if save_dir is not None:
            ensure_dir(save_dir)
            metric_as_string_no_spaces = metric_as_string.replace(" ", "_")
            basename = f"{metric_as_string_no_spaces}_show_CB_logit={show_cb_logit}_legend={add_legend_to_plot}.png"
            filepath = os.path.join(save_dir, basename)
            plt.savefig(filepath, bbox_inches="tight")
            if save_legend_separately and m == len(list_of_metrics_as_strings) - 1:
                legend = plt.legend(
                    shadow=True,
                    fancybox=True,
                    bbox_to_anchor=(1, 0.5, 0.3, 0.2),
                    borderaxespad=0,
                )
                fig = legend.figure
                fig.canvas.draw()
                bbox = legend.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )
                basename = f"legend_only_show_CB_logit={show_cb_logit}.png"
                filepath = os.path.join(save_dir, basename)
                fig.savefig(filepath, dpi="figure", bbox_inches=bbox)
        else:
            plt.show()
        plt.clf()


def plot_performance_over_time_results(
    performance_over_time_results: PerformanceOverTimeResults,
    save_dir: Optional[str] = None,
    log_like_data_generating_process: Optional[float] = None,
    accuracy_data_generating_process: Optional[float] = None,
    log_like_random_guessing: Optional[float] = None,
    accuracy_random_guessing: Optional[float] = None,
    min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve=0.0,
    min_log_likelihood_for_y_axis: Optional[float] = None,
    max_log_likelihood_for_y_axis: Optional[float] = None,
    add_legend_to_plot: bool = True,
    show_cb_logit: bool = True,
    CBC_name: str = "CBC",
    CBM_name: str = "CBM",
):
    # convert advi_results_by_lr to df_performance_advi_by_lr
    # then we can feed `plot_performance_over_time` a DataFrame (or dict of DataFrames)
    # for each inference method
    df_performance_advi_by_lr = {}
    for lr, advi_results in performance_over_time_results.advi_results_by_lr.items():
        df_performance_advi_by_lr[lr] = advi_results.performance_ADVI

    return plot_performance_over_time(
        df_performance_advi_by_lr,
        performance_over_time_results.df_performance_cavi_probit,
        performance_over_time_results.df_performance_cavi_logit,
        performance_over_time_results.df_performance_nuts,
        performance_over_time_results.df_performance_softmax_via_pga_and_gibbs,
        save_dir,
        log_like_data_generating_process,
        accuracy_data_generating_process,
        log_like_random_guessing,
        accuracy_random_guessing,
        min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve,
        min_log_likelihood_for_y_axis,
        max_log_likelihood_for_y_axis,
        add_legend_to_plot,
        show_cb_logit,
        CBC_name,
        CBM_name,
    )
