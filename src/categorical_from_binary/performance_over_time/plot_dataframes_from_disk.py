import os
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from pandas.core.frame import DataFrame

from categorical_from_binary.io import ensure_dir, read_json
from categorical_from_binary.performance_over_time.metadata import MetaData
from categorical_from_binary.performance_over_time.plotter import (
    plot_performance_over_time,
)


def _get_pandas_dataframe_or_return_none(
    results_dir: str, dir_tail: str
) -> Optional[DataFrame]:
    if dir_tail is None:
        return None
    else:
        return pd.read_csv(os.path.join(results_dir, dir_tail))


def make_performance_over_time_plots_from_dataframes_on_disk(
    results_dir: str,
    dir_tail_to_cavi_probit: Optional[str],
    dir_tail_to_cavi_logit: Optional[str],
    dir_tail_to_nuts: Optional[str],
    dir_tail_to_gibbs: Optional[str],
    dir_tails_to_advi: Optional[Dict[float, str]],
    dir_tail_for_writing_plots: Optional[str] = "plots/",
    min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve: Optional[float] = 0.0,
    min_log_likelihood_for_y_axis: Optional[Union[float, str]] = None,
    max_log_likelihood_for_y_axis: Optional[float] = None,
    CBC_name: str = "CBC",
    CBM_name: str = "CBM",
    SOFTMAX_name: str = "MULTI_LOGIT_NON_IDENTIFIED",
    nuts_link_name: str = "MULTI_LOGIT_NON_IDENTIFIED",
    nuts_link_name_formatted_for_legend: str = "Softmax",
):
    """

    Plot performance over time results based on performance over time dataframes
    that are stored on disk.

    Usage:

    ### first specify paths to dataframes
    RESULTS_DIR="/data/results/arxiv_prep/cluster/larger_sims/"

    dir_tail_to_cavi_probit="04_29_2022_15_27_05_MDT_ONLY_CAVI_PROBIT/result_data_frames/perf_cavi_probit.csv"
    dir_tail_to_gibbs="05_04_2022_08_18_44_MDT_ONLY_SOFTMAX_VIA_PGA_AND_GIBBS/result_data_frames/perf_softmax_via_pga_and_gibbs.csv"
    dir_tail_to_nuts="05_01_2022_17_56_19_MDT_ONLY_NUTS/result_data_frames/perf_nuts.csv"
    dir_tails_to_advi={
        0.1: "04_30_2022_00_46_30_MDT_ONLY_ADVI/result_data_frames/perf_advi_0.1.csv",
        0.01: "04_30_2022_00_46_30_MDT_ONLY_ADVI/result_data_frames/perf_advi_0.01.csv",
        0.001: "05_04_2022_00_58_34_MDT_ONLY_ADVI/result_data_frames/perf_advi_0.001.csv",
    }
    dir_tail_to_cavi_logit = None

    # plot configs
    dir_tail_for_writing_plots = "tmp/"
    min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve = 0.5
    min_log_likelihood_for_y_axis = "random guessing"
    max_log_likelihood_for_y_axis = None

    make_performance_over_time_plots_from_dataframes_on_disk(
        RESULTS_DIR,
        dir_tail_to_cavi_probit,
        dir_tail_to_cavi_logit,
        dir_tail_to_nuts,
        dir_tail_to_gibbs,
        dir_tails_to_advi,
        dir_tail_for_writing_plots,
        min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve,
        min_log_likelihood_for_y_axis,
        max_log_likelihood_for_y_axis,
    )
    """

    ### now load in the dataframes (or None if dir tails given as none)
    df_performance_cavi_probit = _get_pandas_dataframe_or_return_none(
        results_dir, dir_tail_to_cavi_probit
    )
    df_performance_cavi_logit = _get_pandas_dataframe_or_return_none(
        results_dir, dir_tail_to_cavi_logit
    )
    df_performance_nuts = _get_pandas_dataframe_or_return_none(
        results_dir, dir_tail_to_nuts
    )
    df_performance_softmax_via_pga_and_gibbs = _get_pandas_dataframe_or_return_none(
        results_dir, dir_tail_to_gibbs
    )

    lrs = sorted(dir_tails_to_advi.keys())
    df_performance_advi_by_lr = {}
    for lr in lrs:
        df_performance_advi_by_lr[lr] = _get_pandas_dataframe_or_return_none(
            results_dir, dir_tails_to_advi[lr]
        )

    ### now load in metadata
    # the metadata lives in all directories, here we take it from probit.
    dir_with_probit = str(Path(dir_tail_to_cavi_probit).parent.parent)
    dir_tail_to_metadata = f"{dir_with_probit}/metadata.json"
    metadata_as_dict = read_json(os.path.join(results_dir, dir_tail_to_metadata))
    metadata = MetaData(**metadata_as_dict)

    ### now plot
    plot_dir = os.path.join(results_dir, dir_tail_for_writing_plots)
    ensure_dir(plot_dir)

    for show_cb_logit in [True, False]:
        for add_legend_to_plot in [True, False]:
            save_legend_separately = not add_legend_to_plot
            label_advi_lrs_by_index = save_legend_separately
            plot_performance_over_time(
                df_performance_advi_by_lr,
                df_performance_cavi_probit,
                df_performance_cavi_logit,
                df_performance_nuts,
                df_performance_softmax_via_pga_and_gibbs,
                plot_dir,
                metadata.mean_log_like_data_generating_process,
                metadata.accuracy_data_generating_process,
                metadata.mean_log_like_random_guessing,
                metadata.accuracy_random_guessing,
                min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve=min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve,
                min_log_likelihood_for_y_axis=min_log_likelihood_for_y_axis,
                max_log_likelihood_for_y_axis=max_log_likelihood_for_y_axis,
                add_legend_to_plot=add_legend_to_plot,
                show_cb_logit=show_cb_logit,
                label_advi_lrs_by_index=label_advi_lrs_by_index,
                save_legend_separately=save_legend_separately,
                CBC_name=CBC_name,
                CBM_name=CBM_name,
                SOFTMAX_name=SOFTMAX_name,
                nuts_link_name=nuts_link_name,
                nuts_link_name_formatted_for_legend=nuts_link_name_formatted_for_legend,
            )
