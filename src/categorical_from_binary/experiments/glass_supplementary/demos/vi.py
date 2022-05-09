"""
Goal: 
    Analyze glass identification dataset with CBC-Probit, see if we can match what Johndrow
    et al. got.   Use VI.

Reference:
    You can view the dataset at, e.g., https://github.com/p-sama/Glass-Classification/blob/master/glass.csv
"""

import datetime
import os
import pickle

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.datasets.glass_supplementary.load import (
    construct_glass_identification_data_split,
    load_glass_identification_data,
)
from categorical_from_binary.evaluate.multiclass import (
    take_measurements_comparing_CBM_and_CBC_estimators,
)
from categorical_from_binary.experiments.glass_supplementary.analysis import (
    get_holdout_loglikes,
    get_misclassification_rates,
)
from categorical_from_binary.experiments.glass_supplementary.calibration_analysis import (
    compute_sum_of_squared_calibration_errors_by_link_from_results_on_many_splits,
    plot_CBM_calibration_advantage_curves,
    plot_calibration_curves_for_one_data_split_at_a_time,
)
from categorical_from_binary.experiments.glass_supplementary.plot_vi import (
    make_plot_comparing_holdout_log_likes_for_ib_plus_sdo_vs_ib_plus_do,
    make_violin_plot_of_differences_in_geometric_mean_log_likes_for_ib_plus_sdo_vs_ib_plus_do,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.results import ResultsOnOneSplit


###
# Configs
###

# data construction
pct_training = 0.9
standardize_design_matrix = True
num_data_splits = 10

# VI
convergence_criterion_drop_in_elbo = 0.005

# save to disk
save_to_disk = False
save_dir = "data/cbc_probit/glass_identification/results/"
path_to_last_results = "data/cbc_probit/glass_identification/results/glass_identification_IB_plus_CBM_vs_IB_plus_CBC_with_VI_results_2021-08-09.pkl"
use_last_results_instead_of_running_inference = False

###
# Run the analysis (or, if desired, load last results)
###
if use_last_results_instead_of_running_inference:
    with open(path_to_last_results, "wb") as handle:
        results_on_many_splits = pickle.load(handle)
else:
    glass_identification_data = load_glass_identification_data()

    measurements_on_all_splits = []
    results_on_many_splits = []
    for random_seed in range(num_data_splits):
        print(
            f"\n---- Now running experiment by splitting glass identification data with seed {random_seed} ----"
        )
        split_dataset = construct_glass_identification_data_split(
            glass_identification_data,
            pct_training=pct_training,
            standardize_design_matrix=standardize_design_matrix,
            random_seed=random_seed,
        )

        ####
        # Variational Inference
        ####
        results = compute_multiclass_probit_vi_with_normal_prior(
            split_dataset.labels_train,
            split_dataset.covariates_train,
            variational_params_init=None,
            convergence_criterion_drop_in_elbo=convergence_criterion_drop_in_elbo,
        )
        variational_params = results.variational_params
        beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov
        results_on_one_split = ResultsOnOneSplit(
            random_seed, split_dataset, beta_mean=beta_mean, beta_cov=beta_cov
        )
        results_on_many_splits.append(results_on_one_split)
        ###
        # Evaluate the model quality
        ###
        measurements_on_one_split = take_measurements_comparing_CBM_and_CBC_estimators(
            split_dataset,
            beta_mean,
        )

        measurements_on_all_splits.append(measurements_on_one_split)


###
# Save results (if desired)
###
if save_to_disk:
    todays_date = str(datetime.datetime.now().date())
    results_basename = (
        "glass_identification_IB_plus_CBM_vs_IB_plus_CBC_with_VI_results_"
        + todays_date
        + ".pkl"
    )
    path_to_results = os.path.join(save_dir, results_basename)
    with open(path_to_results, "wb") as handle:
        pickle.dump(results_on_many_splits, handle, protocol=pickle.HIGHEST_PROTOCOL)

###
# Main analysis of results
###

holdout_loglikes = get_holdout_loglikes(measurements_on_all_splits)
make_plot_comparing_holdout_log_likes_for_ib_plus_sdo_vs_ib_plus_do(holdout_loglikes)
make_violin_plot_of_differences_in_geometric_mean_log_likes_for_ib_plus_sdo_vs_ib_plus_do(
    holdout_loglikes
)

misclassification_rates = get_misclassification_rates(measurements_on_all_splits)

###
# Calibration analysis
###

# Compute SSE_calibration
n_bins = 6
sses_by_link = (
    compute_sum_of_squared_calibration_errors_by_link_from_results_on_many_splits(
        results_on_many_splits, n_bins
    )
)
print(
    f"The mean calibration SSEs was {np.mean(sses_by_link[Link.CBC_PROBIT]):.03f} +/- {np.std(sses_by_link[Link.CBC_PROBIT]):.03f}  for VI_IB+CBM and {np.mean(sses_by_link[Link.CBM_PROBIT]):.03f} +/- {np.std(sses_by_link[Link.CBM_PROBIT]):.03f} for VI-IB+CBM"
)

# show calibration curves
plot_calibration_curves_for_one_data_split_at_a_time(results_on_many_splits, n_bins)
plot_CBM_calibration_advantage_curves(results_on_many_splits, n_bins)
