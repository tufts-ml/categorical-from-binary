"""
Goal: compare CBC-Logit to *CBC-Probit 
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from categorical_from_binary.datasets.glass_supplementary.load import (
    construct_glass_identification_data_split,
    load_glass_identification_data,
)
from categorical_from_binary.ib_cavi.multi.inference import (
    IB_Model,
    compute_multiclass_vi_with_normal_prior,
)


###
# Configs
###

# data construction
pct_training = 0.9
standardize_design_matrix = True
num_data_splits = 10

# VI
convergence_criterion_drop_in_mean_elbo = 0.01
# max_n_iterations = 8  # 20

# Notes:
#
# 1) We can't run with `convergence_criterion_drop_in_mean_elbo`, because the ELBO hasn't yet been coded up for
# the CBC-Logit. This is unfortunate because it might take more CAVI iterations for CBC-Logit than CBC-Probit to
# asymptote in quality, due to the fact that the CBC-Logit has a deeper hierarchy (from the polya gamma augmentation).
# So our comparisons that are nominally between CBC-Logit and CBC-Probit might in actuality be comparisons between
# models running for a short vs long time.  Supporting that this might matter, look at the holdout likelihood
# performance as a number of CAVI iterations -- it sometimes goes up and then down a little bit.
#
# 2) We can't do too many CAVI iterations with CBC-Logit, because if we run it for too long, we start to get numerical
# issues. This is probably due to the fact that the CBC-Logit update functions, relative to the CBC-Probit update functions,
# are immature.

###
# Run the analysis (or, if desired, load last results)
###

glass_identification_data = load_glass_identification_data()

results_by_fold = defaultdict(list)

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
    # Variational Inference with multiclass probit regression
    ####
    results_probit = compute_multiclass_vi_with_normal_prior(
        IB_Model.PROBIT,
        split_dataset.labels_train,
        split_dataset.covariates_train,
        labels_test=split_dataset.labels_test,
        covariates_test=split_dataset.covariates_test,
        variational_params_init=None,
        convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
        # max_n_iterations=max_n_iterations,
    )

    ####
    # Variational Inference with multiclass logit regression
    ####
    results_logit = compute_multiclass_vi_with_normal_prior(
        IB_Model.LOGIT,
        split_dataset.labels_train,
        split_dataset.covariates_train,
        labels_test=split_dataset.labels_test,
        covariates_test=split_dataset.covariates_test,
        variational_params_init=None,
        convergence_criterion_drop_in_mean_elbo=convergence_criterion_drop_in_mean_elbo,
        # max_n_iterations=max_n_iterations,
    )

    results_by_fold["CBC_probit_holdout_loglike"].append(
        results_probit.holdout_performance_over_time[
            "mean holdout log likelihood for CBC_PROBIT"
        ].to_numpy()[-1]
    )
    results_by_fold["CBM_probit_holdout_loglike"].append(
        results_probit.holdout_performance_over_time[
            "mean holdout log likelihood for CBM_PROBIT"
        ].to_numpy()[-1]
    )
    results_by_fold["CBC_logit_holdout_loglike"].append(
        results_logit.holdout_performance_over_time[
            "mean holdout log likelihood for CBC_LOGIT"
        ].to_numpy()[-1]
    )
    results_by_fold["CBM_logit_holdout_loglike"].append(
        results_logit.holdout_performance_over_time[
            "mean holdout log likelihood for CBM_LOGIT"
        ].to_numpy()[-1]
    )
    results_by_fold["probit_accuracy"].append(
        results_probit.holdout_performance_over_time[
            "correct classification rate"
        ].to_numpy()[-1]
    )
    results_by_fold["logit_accuracy"].append(
        results_logit.holdout_performance_over_time[
            "correct classification rate"
        ].to_numpy()[-1]
    )


results_by_fold_df = pd.DataFrame(results_by_fold)
mean_holdout_log_likes = results_by_fold_df.mean()[:4]
geometric_mean_holdout_likes = np.exp(mean_holdout_log_likes)
holdout_accuracies = results_by_fold_df.mean()[4:]
print(f"-- Mean holdout log likehoods---\n{mean_holdout_log_likes}", end="\n\n")
print(
    f"---Geometric mean holdout likehoods---\n{geometric_mean_holdout_likes}",
    end="\n\n",
)
print(f"---Holdout accuracies---\n{holdout_accuracies}", end="\n\n")
