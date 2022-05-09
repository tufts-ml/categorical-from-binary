from categorical_from_binary.performance_over_time.plot_dataframes_from_disk import (
    make_performance_over_time_plots_from_dataframes_on_disk,
)


### LARGER SIMS

RESULTS_DIR = "/Users/mwojno01/Repos/categorical_from_binary/data/results/arxiv_prep/cluster/larger_sims/"

dir_tail_to_cavi_probit = (
    "04_29_2022_15_27_05_MDT_ONLY_CAVI_PROBIT/result_data_frames/perf_cavi_probit.csv"
)
dir_tail_to_gibbs = "05_04_2022_08_18_44_MDT_ONLY_SOFTMAX_VIA_PGA_AND_GIBBS/result_data_frames/perf_softmax_via_pga_and_gibbs.csv"
dir_tail_to_nuts = "05_05_2022_08_08_52_MDT_ONLY_NUTS/result_data_frames/perf_nuts.csv"
dir_tails_to_advi = {
    0.1: "04_30_2022_00_46_30_MDT_ONLY_ADVI/result_data_frames/perf_advi_0.1.csv",
    0.01: "04_30_2022_00_46_30_MDT_ONLY_ADVI/result_data_frames/perf_advi_0.01.csv",
    0.001: "05_04_2022_00_58_34_MDT_ONLY_ADVI/result_data_frames/perf_advi_0.001.csv",
}
dir_tail_to_cavi_logit = None

# plot configs
dir_tail_for_writing_plots = "plots/"
min_pct_iterates_with_non_nan_metrics_in_order_to_plot_curve = 0.5
min_log_likelihood_for_y_axis = -10.0
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


### SMALL SIMS

RESULTS_DIR = "/Users/mwojno01/Repos/categorical_from_binary/data/results/arxiv_prep/cluster/small_sims/"

dir_tail_to_cavi_logit = (
    "05_03_2022_08_04_07_MDT_ONLY_CAVI_LOGIT/result_data_frames/perf_cavi_logit.csv"
)
dir_tail_to_cavi_probit = (
    "05_03_2022_08_04_25_MDT_ONLY_CAVI_PROBIT/result_data_frames/perf_cavi_probit.csv"
)
dir_tail_to_nuts = "05_03_2022_08_04_33_MDT_ONLY_NUTS/result_data_frames/perf_nuts.csv"
dir_tail_to_gibbs = "05_03_2022_08_04_39_MDT_ONLY_SOFTMAX_VIA_PGA_AND_GIBBS/result_data_frames/perf_softmax_via_pga_and_gibbs.csv"
dir_tails_to_advi = {
    0.1: "05_03_2022_08_22_46_MDT_ONLY_ADVI/result_data_frames/perf_advi_0.1.csv",
    0.01: "05_03_2022_08_22_46_MDT_ONLY_ADVI/result_data_frames/perf_advi_0.01.csv",
    1.0: "05_03_2022_08_22_46_MDT_ONLY_ADVI/result_data_frames/perf_advi_1.0.csv",
}

# plot configs
dir_tail_for_writing_plots = "plots_next/"
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
