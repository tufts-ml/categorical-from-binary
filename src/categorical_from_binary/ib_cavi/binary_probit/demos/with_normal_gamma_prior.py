import numpy as np

from categorical_from_binary.data_generation.bayes_binary_reg import (
    generate_probit_regression_dataset,
)
from categorical_from_binary.data_generation.util import (
    prepend_features_with_column_of_all_ones_for_intercept,
)
from categorical_from_binary.ib_cavi.binary_probit.inference.main import (
    compute_probit_vi_with_normal_gamma_prior,
)
from categorical_from_binary.ib_cavi.binary_probit.sanity import (
    compute_class_probabilities,
    print_table_comparing_class_probs_to_labels,
)
from categorical_from_binary.selection.core import (
    compute_feature_inclusion_data_frame_using_scaled_neighborhood_method,
)
from categorical_from_binary.selection.evaluate import (
    get_evaluation_result,
    print_report_on_variable_selection_decisions,
)
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
# Create some fake data with sparse betas
###

dataset = generate_probit_regression_dataset(
    n_samples=1000,
    n_features=50,
    n_sparse_features=25,
    n_bounded_from_zero_features=25,
    lower_bound_for_bounded_from_zero_features=0.2,
    seed=1,
)
covariates = prepend_features_with_column_of_all_ones_for_intercept(dataset.features)
labels = dataset.labels

###
# Inference
###

###hyperparameters
# prior
lambda_ = 0.001
variance = 1.0
hyperparameters = hyperparameters_from_lambda_and_desired_marginal_beta_variance(
    variance, lambda_
)

variational_params = compute_probit_vi_with_normal_gamma_prior(
    labels,
    covariates,
    variational_params_init=None,
    max_n_iterations=10,
    convergence_criterion_drop_in_elbo=-np.inf,
    hyperparameters=hyperparameters,
)

variational_beta = variational_params.beta

###
# Sanity checks
###

# ### Sanity check -- show fitted category probabilities for some of the binary responses
print(
    f"Now printing fitted category probabilities for the selected response in a small sample"
)
n_samples = 10
covariates_sample, labels_sample = covariates[:n_samples], labels[:n_samples]
class_probs = compute_class_probabilities(covariates_sample, variational_beta.mean)
print_table_comparing_class_probs_to_labels(labels_sample, class_probs)

input("Press Enter to continue...")

###
# Variable selection
###

neighborhood_probability_threshold_for_exclusion = 0.90
neighborhood_radius_in_units_of_std_devs = 3

beta_mean = variational_params.beta.mean
beta_stds = np.sqrt(np.diag(variational_params.beta.cov))

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
    x_ticks_fontsize=8,
)


evaluation_result_vb = get_evaluation_result(dataset.beta, feature_inclusion_df)

###
# Compare feature selection accuracy to sklearn quickly (i.e. with some preset hyperparameter)
###
sklearn_logistic_lasso_regression_C = 0.04

sklearn_variable_selection_results = (
    get_sklearn_variable_selection_results_using_logistic_regression(
        dataset.features,
        dataset.labels,
        sklearn_logistic_lasso_regression_C,
        multiclass=False,
        penalty="l1",
    )
)
plot_heatmap(
    sklearn_variable_selection_results.beta,
    sklearn_variable_selection_results.feature_inclusion_df,
    white_out_excluded_covariates=True,
    x_ticks_fontsize=8,
)


print_report_on_variable_selection_decisions(
    dataset.features,
    dataset.labels,
    beta_mean,
    beta_stds,
    dataset.beta,
    feature_inclusion_df,
    sklearn_logistic_lasso_regression_C,
    verbose=True,
)

###
# Compare feature selection accuracy to sklearn more properly (with some hyperparameter selection)
###

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
        multiclass=False,
    )

    # TODO: `get_evaluation_results_for_variational_bayes_by_lambdas_` should automatically do the prepending
    # where necessary.
    # NOTE: `get_evaluation_results_for_variational_bayes_by_lambdas_` is slow...
    covariates = prepend_features_with_column_of_all_ones_for_intercept(
        dataset.features
    )
    lambdas_ = np.logspace(start=-5, stop=2.5, num=15, base=10)
    evaluation_results_by_lambdas_ = (
        get_evaluation_results_for_variational_bayes_by_lambdas_(
            dataset.labels,
            covariates,
            dataset.beta,
            multiclass=False,
            lambdas_=lambdas_,
        )
    )
