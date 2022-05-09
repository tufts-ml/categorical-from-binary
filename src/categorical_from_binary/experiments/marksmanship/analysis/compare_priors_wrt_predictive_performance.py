import numpy as np


np.set_printoptions(precision=3, suppress=True)

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.evaluate.multiclass import (
    compute_mean_log_likelihood_for_multiclass_regression,
)
from categorical_from_binary.experiments.marksmanship.preprocess.join import (
    make_marksmanship_data_frame,
)
from categorical_from_binary.experiments.marksmanship.preprocess.util import (
    exclude_based_on_missingness,
    exclude_based_on_zero_std,
)
from categorical_from_binary.experiments.marksmanship.preprocess.xy import (
    add_previous_label_to_covariates,
    add_second_previous_label_to_covariates,
    make_covariates_and_covariate_dict_from_df,
    make_labels_and_label_dict_from_df,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_gamma_prior,
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.selection.hyperparameters import (
    hyperparameters_from_lambda_and_desired_marginal_beta_variance,
)


###
# Configs
###
index_of_previous_label_to_add_to_covariates = 2
threshold_on_proportion_missing_in_columns_for_baseline_data = 0.50
threshold_on_proportion_missing_in_columns_for_merged_data = 0.50
use_baseline_covariates = False

###
# Preprocessing
###

df_all_covariates = make_marksmanship_data_frame(
    use_baseline_covariates,
    threshold_on_proportion_missing_in_columns_for_baseline_data,
)

# handle exclusions
df_all_covariates_without_missing_data = exclude_based_on_missingness(
    df_all_covariates,
    threshold_on_proportion_missing_in_columns_for_merged_data,
)
df = exclude_based_on_zero_std(df_all_covariates_without_missing_data)

(
    labels_before_autoregressive_adjustment,
    label_dict,
) = make_labels_and_label_dict_from_df(df)
(
    covariates_before_autoregressive_adjustment,
    covariate_dict,
) = make_covariates_and_covariate_dict_from_df(df)

if index_of_previous_label_to_add_to_covariates == 0:
    covariates = covariates_before_autoregressive_adjustment
    labels = labels_before_autoregressive_adjustment
elif index_of_previous_label_to_add_to_covariates == 1:
    covariates, covariate_dict, labels = add_previous_label_to_covariates(
        covariates_before_autoregressive_adjustment,
        covariate_dict,
        labels_before_autoregressive_adjustment,
        label_dict,
    )
# Providing the option to add 2 steps back and only 2 steps back is a weird
# hack to account for the fact that the soliders were told to shoot at a target twice.
elif index_of_previous_label_to_add_to_covariates == 2:
    covariates, covariate_dict, labels = add_second_previous_label_to_covariates(
        covariates_before_autoregressive_adjustment,
        covariate_dict,
        labels_before_autoregressive_adjustment,
        label_dict,
    )
else:
    raise ValueError(
        f"I don't understand  {index_of_previous_label_to_add_to_covariates }, the value provided "
        f"for index_of_previous_label_to_add_to_covariates."
    )

# Prep training data
n_train_samples = 7000
covariates_train = covariates[:n_train_samples]
labels_train = labels[:n_train_samples]

###
# Variational inference
###
max_n_iterations = 10

### with normal prior
results_with_normal_prior = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    variational_params_init=None,
    max_n_iterations=max_n_iterations,
)
variational_params_with_normal_prior = results_with_normal_prior.variational_params
beta_mean_with_normal_prior = variational_params_with_normal_prior.beta.mean

### with normal-gamma prior
lambda_ = 0.001
variance = 1.0
hyperparameters = hyperparameters_from_lambda_and_desired_marginal_beta_variance(
    variance, lambda_
)

results_with_normal_gamma_prior = compute_multiclass_probit_vi_with_normal_gamma_prior(
    labels_train,
    covariates_train,
    variational_params_init=None,
    max_n_iterations=max_n_iterations,
    hyperparameters=hyperparameters,
)
variational_params_with_normal_gamma_prior = (
    results_with_normal_gamma_prior.variational_params
)
beta_mean_with_normal_gamma_prior = variational_params_with_normal_gamma_prior.beta.mean


###
# Evaluate predictive performance
###

covariates_test = covariates[n_train_samples:]
labels_test = labels[n_train_samples:]
link_for_category_probabilities = Link.CBM_PROBIT


holdout_log_like_with_normal_prior = (
    compute_mean_log_likelihood_for_multiclass_regression(
        covariates_test,
        labels_test,
        beta_mean_with_normal_prior,
        link_for_category_probabilities,
    )
)
print(
    f"With the normal prior, the holdout log likelihood  is {holdout_log_like_with_normal_prior:.04}"
)

holdout_log_like_with_normal_gamma_prior = (
    compute_mean_log_likelihood_for_multiclass_regression(
        covariates_test,
        labels_test,
        beta_mean_with_normal_gamma_prior,
        link_for_category_probabilities,
    )
)
print(
    f"With the normal-gamma prior, the holdout log likelihood  is {holdout_log_like_with_normal_gamma_prior:.04}"
)
