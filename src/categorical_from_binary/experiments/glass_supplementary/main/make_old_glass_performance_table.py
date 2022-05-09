"""
Goal: Compare performance of 
    * IB-Logit fit with CAVI +CBC/CBM 
    * IB-Probit fit with CAVI +CBC/CBM
    * CBC/CBM-Probit fit with HMC 
    * CBC/CBM-Logit fit with HMC (need to add)
    * MNP fit with HMC (need to add; maybe with aux variable)

Reference:
    You can view the dataset at, e.g., https://github.com/p-sama/Glass-Classification/blob/master/glass.csv
"""

from collections import defaultdict

import pandas as pd

from categorical_from_binary.datasets.glass_supplementary.load import (
    construct_glass_identification_data_split,
    load_glass_identification_data,
)
from categorical_from_binary.experiments.glass_supplementary.hmc_helpers import (
    CategoricalModelType,
    get_accuracy_from_beta_samples,
    get_beta_samples_for_categorical_model_via_HMC,
    get_mean_log_like_from_beta_samples,
)
from categorical_from_binary.ib_cavi.multi.inference import (
    IB_Model,
    compute_multiclass_vi_with_normal_prior,
)
from categorical_from_binary.timing import time_me


###
# Configs
###

# data construction
pct_training = 0.9
standardize_design_matrix = True
num_data_splits = 10

# VI
convergence_criterion_drop_in_mean_elbo = 0.005
# max_n_iterations = 8  # 20

# HMC sampling
num_warmup_samples, num_mcmc_samples = 3000, 10000

# Filepath for saving results
results_filepath = "/Users/mwojno01/Repos/categorical_from_binary/data/results/glass_identification/glass_identification_results.csv"

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

    ####
    # HMC for many models
    ####
    beta_samples_CBC_probit, time_for_CBC_probit_HMC = time_me(
        get_beta_samples_for_categorical_model_via_HMC
    )(
        split_dataset.covariates_train,
        split_dataset.labels_train,
        CategoricalModelType.CBC_PROBIT,
        num_warmup_samples,
        num_mcmc_samples,
    )
    loglike_CBC_probit_HMC = get_mean_log_like_from_beta_samples(
        beta_samples_CBC_probit,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.CBC_PROBIT,
    )
    accuracy_CBC_probit_HMC = get_accuracy_from_beta_samples(
        beta_samples_CBC_probit,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.CBC_PROBIT,
    )

    beta_samples_CBM_probit, time_for_CBM_probit_HMC = time_me(
        get_beta_samples_for_categorical_model_via_HMC
    )(
        split_dataset.covariates_train,
        split_dataset.labels_train,
        CategoricalModelType.CBM_PROBIT,
        num_warmup_samples,
        num_mcmc_samples,
    )
    loglike_CBM_probit_HMC = get_mean_log_like_from_beta_samples(
        beta_samples_CBM_probit,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.CBM_PROBIT,
    )
    accuracy_CBM_probit_HMC = get_accuracy_from_beta_samples(
        beta_samples_CBM_probit,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.CBM_PROBIT,
    )

    beta_samples_CBC_logit, time_for_CBC_logit_HMC = time_me(
        get_beta_samples_for_categorical_model_via_HMC
    )(
        split_dataset.covariates_train,
        split_dataset.labels_train,
        CategoricalModelType.CBC_LOGIT,
        num_warmup_samples,
        num_mcmc_samples,
    )
    loglike_CBC_logit_HMC = get_mean_log_like_from_beta_samples(
        beta_samples_CBC_logit,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.CBC_LOGIT,
    )
    accuracy_CBC_logit_HMC = get_accuracy_from_beta_samples(
        beta_samples_CBC_logit,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.CBC_LOGIT,
    )

    beta_samples_CBM_logit, time_for_CBM_logit_HMC = time_me(
        get_beta_samples_for_categorical_model_via_HMC
    )(
        split_dataset.covariates_train,
        split_dataset.labels_train,
        CategoricalModelType.CBM_LOGIT,
        num_warmup_samples,
        num_mcmc_samples,
    )
    loglike_CBM_logit_HMC = get_mean_log_like_from_beta_samples(
        beta_samples_CBM_logit,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.CBM_LOGIT,
    )
    accuracy_CBM_logit_HMC = get_accuracy_from_beta_samples(
        beta_samples_CBM_logit,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.CBM_LOGIT,
    )

    beta_samples_softmax, time_for_softmax_HMC = time_me(
        get_beta_samples_for_categorical_model_via_HMC
    )(
        split_dataset.covariates_train,
        split_dataset.labels_train,
        CategoricalModelType.SOFTMAX,
        num_warmup_samples,
        num_mcmc_samples,
    )
    loglike_softmax_HMC = get_mean_log_like_from_beta_samples(
        beta_samples_softmax,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.SOFTMAX,
    )
    accuracy_softmax_HMC = get_accuracy_from_beta_samples(
        beta_samples_softmax,
        split_dataset.covariates_test,
        split_dataset.labels_test,
        CategoricalModelType.SOFTMAX,
    )

    results_by_fold["holdout_loglike_IB_Probit_plus_CBC"].append(
        results_probit.holdout_performance_over_time[
            "mean holdout log likelihood for CBC_PROBIT"
        ].to_numpy()[-1]
    )
    results_by_fold["holdout_loglike_IB_Probit_plus_CBM"].append(
        results_probit.holdout_performance_over_time[
            "mean holdout log likelihood for CBM_PROBIT"
        ].to_numpy()[-1]
    )
    results_by_fold["holdout_loglike_IB_Logit_plus_CBC"].append(
        results_logit.holdout_performance_over_time[
            "mean holdout log likelihood for CBC_LOGIT"
        ].to_numpy()[-1]
    )
    results_by_fold["holdout_loglike_IB_Logit_plus_CBM"].append(
        results_logit.holdout_performance_over_time[
            "mean holdout log likelihood for CBM_LOGIT"
        ].to_numpy()[-1]
    )
    results_by_fold["holdout_loglike_CBC_Probit_HMC"].append(loglike_CBC_probit_HMC)
    results_by_fold["holdout_loglike_CBM_Probit_HMC"].append(loglike_CBM_probit_HMC)
    results_by_fold["holdout_loglike_CBC_Logit_HMC"].append(loglike_CBC_logit_HMC)
    results_by_fold["holdout_loglike_CBM_Logit_HMC"].append(loglike_CBM_logit_HMC)
    results_by_fold["holdout_loglike_softmax_HMC"].append(loglike_softmax_HMC)
    results_by_fold["accuracy_IB_Probit_CAVI"].append(
        results_probit.holdout_performance_over_time[
            "correct classification rate"
        ].to_numpy()[-1]
    )
    results_by_fold["accuracy_IB_Logit_CAVI"].append(
        results_logit.holdout_performance_over_time[
            "correct classification rate"
        ].to_numpy()[-1]
    )
    results_by_fold["accuracy_CBC_probit_HMC"].append(accuracy_CBC_probit_HMC)
    results_by_fold["accuracy_CBM_probit_HMC"].append(accuracy_CBM_probit_HMC)
    results_by_fold["accuracy_CBC_logit_HMC"].append(accuracy_CBC_probit_HMC)
    results_by_fold["accuracy_CBM_logit_HMC"].append(accuracy_CBM_probit_HMC)
    results_by_fold["accuracy_softmax_HMC"].append(accuracy_softmax_HMC)
    results_by_fold["time_for_IB_Logit"].append(
        results_logit.holdout_performance_over_time[
            "seconds elapsed (cavi)"
        ].to_numpy()[-1]
    )
    results_by_fold["time_for_IB_Probit"].append(
        results_probit.holdout_performance_over_time[
            "seconds elapsed (cavi)"
        ].to_numpy()[-1]
    )
    results_by_fold["time_for_CBC_Probit_HMC"].append(time_for_CBC_probit_HMC)
    results_by_fold["time_for_CBM_Probit_HMC"].append(time_for_CBM_probit_HMC)
    results_by_fold["time_for_CBC_Logit_HMC"].append(time_for_CBC_logit_HMC)
    results_by_fold["time_for_CBM_Logit_HMC"].append(time_for_CBM_logit_HMC)
    results_by_fold["time_for_softmax_HMC"].append(time_for_softmax_HMC)

results_by_fold_df = pd.DataFrame(results_by_fold)
pd.options.display.max_columns = None
print(results_by_fold_df)
results_by_fold_df.to_csv(results_filepath)

# print(results_by_fold_df.to_latex(float_format="%.2f", index=False))
# print(results_by_fold_df.mean().to_latex(float_format="%.2f", index=False))

# Display average results:
print(f"-- Average results---\n{results_by_fold_df.mean()}", end="\n\n")
