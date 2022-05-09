from pprint import pprint

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.data_generation.hierarchical_multiclass_reg import (
    generate_hierarchical_multiclass_regression_dataset,
)
from categorical_from_binary.data_generation.splitter import (
    split_hierarchical_multiclass_regression_dataset,
)
from categorical_from_binary.evaluate.hierarchical_multiclass import (
    compute_metrics_for_hierarchical_multiclass_regression,
    find_rarest_category,
)
from categorical_from_binary.evaluate.multiclass import (
    compute_mean_likelihood_for_multiclass_regression,
)
from categorical_from_binary.ib_cavi.multi.hierarchical_ib_probit.inference import (
    data_dimensions_from_hierarchical_dataset,
    run_hierarchical_multiclass_probit_vi,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)


raise NotImplementedError(
    "This module needs updating -- the metrics have gotten out of data."
)

###
#  A dataset where the first category is pretty rare AND there is a predictive covariate.
###

print(
    f"\nA hierarchical categorical dataset where the first category is pretty rare "
    f"BUT there is a highly predictive covariate."
)


###
#  A dataset where the first category is pretty rare AND there is a predictive covariate.
###

print(
    f"\nA hierarchical categorical dataset where the first category is pretty rare "
    f"BUT there is a highly predictive covariate."
)

seed = 123456

n_samples = 1000
n_train_samples = 50
n_categories = 5
n_groups = 10

# intercept info
include_intercept = True
beta_0_expected = np.array([-2.0, 0.0, 0.0, 0.0, 0.0])

# exogenous feature info
n_features_exogenous = 3
beta_binary_feature = np.array([4.0, 0.0, 0.0, 0.0, 0.0])
beta_exogenous_expected = np.vstack(
    (beta_binary_feature, np.random.normal(scale=1, size=(2, 5)))
)
indices_of_exogenous_features_that_are_binary = [0]
success_probs_for_exogenous_features_that_are_binary = [0.05]

# autoregressive info
is_autoregressive = True
beta_transition_expected = (
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    * 0.1
)


# variances
s2_beta_expected = 0.1
s2_beta = 0.1

hd = generate_hierarchical_multiclass_regression_dataset(
    n_samples,
    n_features_exogenous,
    n_categories,
    n_groups,
    include_intercept=include_intercept,
    is_autoregressive=is_autoregressive,
    s2_beta_expected=s2_beta_expected,
    s2_beta=s2_beta,
    beta_0_expected=beta_0_expected,
    beta_exogenous_expected=beta_exogenous_expected,
    beta_transition_expected=beta_transition_expected,
    indices_of_exogenous_features_that_are_binary=indices_of_exogenous_features_that_are_binary,
    success_probs_for_exogenous_features_that_are_binary=success_probs_for_exogenous_features_that_are_binary,
    seed=seed,
)

# split into train and test sets
hd_train, hd_test = split_hierarchical_multiclass_regression_dataset(
    hd, n_train_samples
)

# find rarest category
rarest_category_in_training_dataset = find_rarest_category(hd_train)


###
# Inference with hierarchical model
###

convergence_criterion_drop_in_elbo_for_hierarchical_multiclass_model = 0.1
variational_params = run_hierarchical_multiclass_probit_vi(
    hd_train,
    convergence_criterion_drop_in_elbo=convergence_criterion_drop_in_elbo_for_hierarchical_multiclass_model,
)

###
# Inference with separate flat models
###


convergence_criterion_drop_in_elbo_for_separate_multiclass_models = 0.1
beta_means_by_group = [None] * n_groups
beta_covs_by_group = [None] * n_groups
for j in range(n_groups):
    results = compute_multiclass_probit_vi_with_normal_prior(
        hd_train.datasets[j].labels,
        hd_train.datasets[j].features,
        convergence_criterion_drop_in_elbo=convergence_criterion_drop_in_elbo_for_separate_multiclass_models,
    )
    variational_params = results.variational_params
    beta_mean, beta_cov = variational_params.beta.mean, variational_params.beta.cov
    beta_means_by_group[j] = beta_mean
    beta_covs_by_group[j] = beta_cov


###
# Metrics with hierarchical model
###

metrics_hierarchical = compute_metrics_for_hierarchical_multiclass_regression(
    variational_params,
    hd_test,
    rarest_category_in_training_dataset,
)
mean_choice_probs_for_hierarchical_model = (
    metrics_hierarchical.mean_choice_probs_by_group
)
mean_rare_category_probs_for_hierarchical_model = (
    metrics_hierarchical.mean_choice_probs_by_group_for_rarest_category
)

###
# Metrics with separate flat models
###

mean_choice_probs_for_separate_models = [None] * n_groups
mean_choice_probs_for_rarest_category_for_separate_models = [None] * n_groups


for j in range(n_groups):
    mean_choice_probs_for_separate_models[
        j
    ] = compute_mean_likelihood_for_multiclass_regression(
        hd_test.datasets[j].features,
        hd_test.datasets[j].labels,
        beta_means_by_group[j],
        model_link=Link.CBC_PROBIT,
    )
    mean_choice_probs_for_rarest_category_for_separate_models[
        j
    ] = compute_mean_likelihood_for_multiclass_regression(
        hd_test.datasets[j].features,
        hd_test.datasets[j].labels,
        beta_means_by_group[j],
        model_link=Link.CBC_PROBIT,
        focal_choice=rarest_category_in_training_dataset,
    )


####
# Report on Metrics
####
J, M, K, Ns = data_dimensions_from_hierarchical_dataset(hd)
print(
    f"\nNum groups: {J}"
    f"\nMean number of observations per group: {np.mean(Ns):.04}"
    f"\nNum categories: {K}"
    f"\nNum designed features: {M}"
)

print("Expected beta")
print(hd.beta_expected)

category_counts_per_sequence = [sum(dataset.labels, 0) for dataset in hd_train.datasets]
print("Category counts per sequence in training set")
pprint(category_counts_per_sequence)

print(f"\nFor {n_groups} separate models, the mean choice probs were: ")
print([f"{p:.03}" for p in mean_choice_probs_for_separate_models])

print(f"\nFor the hierarchical model, the mean choice probs were: ")
print([f"{p:.03}" for p in mean_choice_probs_for_hierarchical_model])

print(
    f"\nFor {n_groups} separate models, the mean choice probs for the RAREST category were: "
)
print([f"{p:.03}" for p in mean_choice_probs_for_rarest_category_for_separate_models])

print(
    f"\nFor the hierarchical model, the mean choice probs for the RAREST category were: "
)
print([f"{p:.03}" for p in mean_rare_category_probs_for_hierarchical_model])


"""
BELOW I SAVE RESULTS FROM SOME INTERESTING RUNS
"""

"""


###
#  A dataset where the first category is pretty rare AND there is a predictive covariate.
###

print(
    f"\nA hierarchical categorical dataset where the first category is pretty rare "
    f"BUT there is a highly predictive covariate."
)

seed = 123456

n_samples = 1000
n_train_samples = 50
n_categories = 5
n_groups = 10

# intercept info
include_intercept = True
beta_0_expected = np.array([-2.0, 0.0, 0.0, 0.0, 0.0])

# exogenous feature info
n_features_exogenous = 3
beta_binary_feature = np.array([4.0, 0.0, 0.0, 0.0, 0.0])
beta_exogenous_expected = np.vstack(
    (beta_binary_feature, np.random.normal(scale=1, size=(2, 5)))
)
indices_of_exogenous_features_that_are_binary = [0]
success_probs_for_exogenous_features_that_are_binary = [0.05]

# autoregressive info
is_autoregressive = True
beta_transition_expected = (
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    * 0.1
)


# variances
s2_beta_expected = 0.1
s2_beta = 0.1

Num groups: 10
Mean number of observations per group: 1e+03
Num categories: 5
Num designed features: 9
Expected beta
[[-2.          0.          0.          0.          0.        ]
 [ 4.          0.          0.          0.          0.        ]
 [-0.34785064 -0.23815091  2.18435137 -0.2511271  -0.12041376]
 [ 1.41138467 -0.25438411  0.96862363  0.93614346 -1.33991502]
 [ 0.          0.          0.          0.          0.        ]
 [ 0.          0.1         0.          0.          0.        ]
 [ 0.          0.          0.1         0.          0.        ]
 [ 0.          0.          0.          0.1         0.        ]
 [ 0.          0.          0.          0.          0.1       ]]
Category counts per sequence in training set
[array([ 6,  9, 10,  5, 20]),
 array([ 3,  5, 23,  5, 14]),
 array([ 2,  2, 17, 12, 17]),
 array([ 2,  5, 14, 14, 15]),
 array([ 6, 13, 18,  6,  7]),
 array([ 2, 11, 17, 14,  6]),
 array([ 1,  1, 13, 20, 15]),
 array([ 4,  4, 20,  8, 14]),
 array([ 4,  1, 11, 11, 23]),
 array([ 2,  7, 20, 12,  9])]

For 10 separate models, the mean choice probs were: 
['0.555', '0.637', '0.629', '0.568', '0.565', '0.498', '0.736', '0.646', '0.58', '0.576']

For the hierarchical model, the mean choice probs were: 
['0.585', '0.657', '0.642', '0.594', '0.584', '0.526', '0.762', '0.679', '0.621', '0.604']

For 10 separate models, the mean choice probs for the RAREST category were: 
['0.388', '0.288', '0.194', '0.0659', '0.482', '0.192', '0.092', '0.444', '0.431', '0.264']

For the hierarchical model, the mean choice probs for the RAREST category were: 
['0.504', '0.503', '0.47', '0.674', '0.59', '0.403', '0.525', '0.706', '0.674', '0.585']

"""
