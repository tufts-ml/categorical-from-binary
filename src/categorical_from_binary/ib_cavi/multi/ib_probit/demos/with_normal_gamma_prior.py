"""
The model is the CBC-Probit model.
Here we do coordinate ascent variational inference (CAVI)

We look at the variable selection decisions
"""

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_gamma_prior,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.util import beta_stds_from_beta_cov
from categorical_from_binary.selection.core import (
    compute_feature_inclusion_data_frame_using_scaled_neighborhood_method,
)
from categorical_from_binary.selection.evaluate import get_evaluation_result
from categorical_from_binary.selection.hyperparameters import (
    hyperparameters_from_lambda_and_desired_marginal_beta_variance,
)
from categorical_from_binary.selection.roc import (
    get_evaluation_results_for_variational_bayes_by_lambdas_,
)
from categorical_from_binary.selection.sklearn import (
    get_evaluation_results_by_sklearn_Cs,
    get_sklearn_variable_selection_results_using_logistic_regression,
)
from categorical_from_binary.show.heatmap import plot_heatmap


###
# Some simulation-level parameters
###

make_comparison_with_hyperparameter_selection = False


###
# Construct dataset
###
n_samples = 5000
n_categories = 4
n_features = 50
n_sparse_features = 25
n_bounded_from_zero_features = 25
lower_bound_for_bounded_from_zero_features = 0.2
include_intercept = True
link = (
    Link.CBM_PROBIT
)  # Note: don't know how to assess variable inclusion accuracy for MULTI-LOGIT (which has 3 betas)
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    n_sparse_features=n_sparse_features,
    n_bounded_from_zero_features=n_bounded_from_zero_features,
    lower_bound_for_bounded_from_zero_features=lower_bound_for_bounded_from_zero_features,
    beta_0=None,
    link=link,
    seed=None,
    include_intercept=include_intercept,
)


# Prep training data
n_train_samples = 4000
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
# `labels_train` gives one-hot encoded representation of category

####
# Variational Inference
####

###hyperparameters
# prior
lambda_ = 1.0
variance = 1.0
hyperparameters = hyperparameters_from_lambda_and_desired_marginal_beta_variance(
    variance, lambda_
)

max_n_iterations = 10
results = compute_multiclass_probit_vi_with_normal_gamma_prior(
    labels_train,
    covariates_train,
    variational_params_init=None,
    max_n_iterations=max_n_iterations,
    hyperparameters=hyperparameters,
)
variational_params = results.variational_params
beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov
beta_stds = beta_stds_from_beta_cov(beta_cov)

####
# Evaluate model selection
####

### plot model
neighborhood_probability_threshold_for_exclusion = 0.90
neighborhood_radius_in_units_of_std_devs = 3
feature_inclusion_df = (
    compute_feature_inclusion_data_frame_using_scaled_neighborhood_method(
        beta_mean,
        beta_stds,
        neighborhood_probability_threshold_for_exclusion,
        neighborhood_radius_in_units_of_std_devs,
    )
)
plot_heatmap(
    beta_mean,
    feature_inclusion_df,
    white_out_excluded_covariates=True,
)


evaluation_result_cavi = get_evaluation_result(dataset.beta, feature_inclusion_df)
print(f"CAVI evaluation: {evaluation_result_cavi}")
input("Press any key to continue")

### compare to sklearn
sklearn_logistic_lasso_regression_C = 0.014

sklearn_variable_selection_results = (
    get_sklearn_variable_selection_results_using_logistic_regression(
        dataset.features,
        dataset.labels,
        sklearn_logistic_lasso_regression_C,
        multiclass=True,
        penalty="l1",
    )
)
plot_heatmap(
    sklearn_variable_selection_results.beta,
    sklearn_variable_selection_results.feature_inclusion_df,
    white_out_excluded_covariates=True,
)

evaluation_result_sklearn = get_evaluation_result(
    dataset.beta, sklearn_variable_selection_results.feature_inclusion_df
)
print(f"Sklearn evaluation: {evaluation_result_sklearn}")
input("Press any key to continue")

###
# Compare feature selection accuracy to sklearn more properly (with some hyperparameter selection)
###
# Below here is a WIP

if make_comparison_with_hyperparameter_selection:
    # To compare to sklearn more properly, we might want to equate them by finding a good hyperparameter for both.
    # TODO: Make the hyperparameter selection automatic in the code. Currently I'm finding this manually
    # by inspecting the results of the two dictionaries returned below.   Warning, the below might take a while.
    sklearn_logistic_lasso_regression_Cs = np.logspace(
        start=-2, stop=0, num=24, base=10
    )
    evaluation_results_by_sklearn_Cs = get_evaluation_results_by_sklearn_Cs(
        dataset.features,
        dataset.labels,
        dataset.beta,
        sklearn_logistic_lasso_regression_Cs,
        multiclass=True,
    )
    print(evaluation_results_by_sklearn_Cs)

    # TODO: `get_evaluation_results_for_variational_bayes_by_lambdas_` should automatically do the prepending
    # where necessary.
    # NOTE: `get_evaluation_results_for_variational_bayes_by_lambdas_` is slow...
    covariates = dataset.features
    lambdas_ = np.logspace(start=-5, stop=2.5, num=15, base=10)
    evaluation_results_by_lambdas_ = (
        get_evaluation_results_for_variational_bayes_by_lambdas_(
            dataset.labels,
            covariates,
            dataset.beta,
            multiclass=True,
            max_n_iterations=max_n_iterations,
            lambdas_=lambdas_,
        )
    )
