"""
In this script, we explore the properties of our code in inference.py which does
variational Inference for Bayesian multiclass logistic regression with 
* stick-breaking link function
* polya-gamma augmentation (for conditional conjugacy)

In particular, as of 3/15/2021, our demo reveals that the code is not functioning
as well as we would have hoped, so we try to understand why.
"""

import copy

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    construct_multi_logit_probabilities,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.evaluate.multiclass import (
    get_sklearn_category_probabilities,
)
from categorical_from_binary.polya_gamma.stickbreak_multiclass_logreg_vi.inference import (
    PriorParameters_MulticlassLogisticRegression,
    VariationalParameters_MulticlassLogisticRegression,
    run_polya_gamma_variational_inference_for_bayesian_multiclass_logistic_regression,
)


dataset = generate_multiclass_regression_dataset(
    n_samples=10000, n_features=5, n_categories=4, seed=10
)

"""
Q: Does the problem go away if I initialize to the truth ?
"""

# strategy: replace the initialization lines in the code with this.
# TODO: Make it possible to initialize from the function call, with optional argument
# that if set to None defaults to the old strategy.
#
# variational_mean_for_betas[0] = dataset.beta[:, 0]
# variational_mean_for_betas[1] = dataset.beta[:, 1]
# variational_cov_for_betas = copy.copy(prior_covs_by_category)
# n_iterations = 30

"""
A: Nope.. first coordinate (first column of beta matrix, corresponding to the first category) 
still slips off to junk... And it does so almost instantly (on first iteration of variational inference)
"""

"""
Q: Does the problem go away if I use the Linderman initialization to correct for label assymetry?
"""

beta_dim_per_category, n_free_categories = np.shape(dataset.beta)
prior_means_beta = [np.zeros(beta_dim_per_category)] * n_free_categories
prior_covs_beta = [np.eye(beta_dim_per_category)] * n_free_categories
prior_params = PriorParameters_MulticlassLogisticRegression(
    prior_means_beta, prior_covs_beta
)

# variational initialization
# here we try to use Linderman's initialization to correct for label assymmetry
# see lines 363-364 of https://github.com/slinderman/pypolyagamma/blob/1f02f7f15eec0e6b7813ed8dc21e66ab2a1b2305/pypolyagamma/distributions.py
from pypolyagamma.utils import compute_psi_cmoments


n_categories = n_free_categories + 1
bias_correction, _ = compute_psi_cmoments(np.ones(n_categories))
variational_params_init = VariationalParameters_MulticlassLogisticRegression(
    means_beta=[
        [bias_correction[k]] + [0] * (beta_dim_per_category - 1)
        for k in range(n_free_categories)
    ],
    covs_beta=copy.deepcopy(prior_covs_beta),
)


variational_params = (
    run_polya_gamma_variational_inference_for_bayesian_multiclass_logistic_regression(
        dataset,
        prior_params,
        variational_params_init,
        max_n_iterations=20,
        verbose=True,
    )
)

### compare betas
print(f"Comparing true beta to variational mean")
print(f"true: {dataset.beta}")
print(f"variational: {np.transpose(variational_params.means_beta)}")


"""
A: Nope
"""

"""
Q: Does the problem go away if I use a weaker prior?
"""

beta_dim_per_category, n_free_categories = np.shape(dataset.beta)
prior_means_beta = [np.zeros(beta_dim_per_category)] * n_free_categories
COVARIANCE_INFLATION_FACTOR = 1000  # pick it high to have less smoothing to prior.
prior_covs_beta = [
    np.eye(beta_dim_per_category) * COVARIANCE_INFLATION_FACTOR
] * n_free_categories
prior_params = PriorParameters_MulticlassLogisticRegression(
    prior_means_beta, prior_covs_beta
)


variational_params = (
    run_polya_gamma_variational_inference_for_bayesian_multiclass_logistic_regression(
        dataset,
        prior_params,
        variational_params_init=None,
        max_n_iterations=20,
        verbose=True,
    )
)

### compare betas
print(f"Comparing true beta to variational mean")
print(f"true: {dataset.beta}")
print(f"variational: {np.transpose(variational_params.means_beta)}")

"""
A: Nope. 
"""


"""
Q: Can we alter things to make our scheme look more like scikitlearn?
"""

# e.g. maybe the scikitlearn has nice regularization properties; if we turn it off we'll look more like theirs

n_categories = 4
ORDER_CATEGORIES_BY_ASCENDING_POPULARITY = True  # set to true for better results
beta_0_for_ordering_categories_from_least_to_most_popular = [
    -3,
    -2,
    -1,
]  # there is an implied 0 for the last category
beta_0_for_ordering_categories_from_most_to_least_popular = [
    3,
    2,
    1,
]  # there is an implied 0 for the last category
if ORDER_CATEGORIES_BY_ASCENDING_POPULARITY:
    beta_0 = beta_0_for_ordering_categories_from_least_to_most_popular
else:
    beta_0 = beta_0_for_ordering_categories_from_most_to_least_popular
dataset = generate_multiclass_regression_dataset(
    n_samples=10000, n_features=5, n_categories=n_categories, beta_0=beta_0, seed=30
)
print(f"The number of times each category label was used: {np.sum(dataset.labels,0)}")

COVARIANCE_INFLATION_FACTOR = 5  # pick it high to have less smoothing to prior.
beta_dim_per_category, n_free_categories = np.shape(dataset.beta)
prior_means_beta = [np.zeros(beta_dim_per_category)] * n_free_categories
prior_covs_beta = [
    np.eye(beta_dim_per_category) * COVARIANCE_INFLATION_FACTOR
] * n_free_categories
prior_params = PriorParameters_MulticlassLogisticRegression(
    prior_means_beta, prior_covs_beta
)

variational_params = (
    run_polya_gamma_variational_inference_for_bayesian_multiclass_logistic_regression(
        dataset,
        prior_params,
        variational_params_init=None,
        max_n_iterations=20,
        verbose=True,
    )
)

### compare betas
print(f"Comparing true beta to variational mean")
print(f"true: \n {dataset.beta}")
print(f"variational: \n {np.transpose(variational_params.means_beta)}")

### compare category probs
true_category_probs = construct_multi_logit_probabilities(
    dataset.features, dataset.beta
)
variational_category_probs = construct_multi_logit_probabilities(
    dataset.features, np.transpose(variational_params.means_beta)
)
sklearn_category_probs = get_sklearn_category_probabilities(dataset, penalty="none")

np.set_printoptions(precision=3, suppress=True)
print("Category probs (training set)")
print(f"true: \n {true_category_probs}")
print(f"sklearn: \n {sklearn_category_probs}")
print(f"variational: \n {variational_category_probs}")

# crappy error comparisons, this isn't respecting the fact that the rows are on the simplex
sum_absolute_error_sklearn = np.sum(abs(true_category_probs - sklearn_category_probs))
sum_absolute_error_variational = np.sum(
    abs(true_category_probs - variational_category_probs)
)
print("Sum of absolute error in (training set) category probs")
print(f"sklearn: \n {sum_absolute_error_sklearn}")
print(f"variational: \n {sum_absolute_error_variational}")


"""
Q:  What happens in the simpler case where we only have beta_0's
"""
dataset = generate_multiclass_regression_dataset(
    n_samples=10000, n_features=0, n_categories=4, seed=30
)
print(f"The number of times each category label was used: {np.sum(dataset.labels,0)}")

beta_dim_per_category, n_free_categories = np.shape(dataset.beta)
prior_means_beta = [np.zeros(beta_dim_per_category)] * n_free_categories
prior_covs_beta = [np.eye(beta_dim_per_category)] * n_free_categories
prior_params = PriorParameters_MulticlassLogisticRegression(
    prior_means_beta, prior_covs_beta
)

variational_params = (
    run_polya_gamma_variational_inference_for_bayesian_multiclass_logistic_regression(
        dataset,
        prior_params,
        variational_params_init=None,
        max_n_iterations=20,
        verbose=True,
    )
)

### compare betas
print(f"Comparing true beta to variational mean")
print(f"true: \n {dataset.beta}")
print(f"variational: \n {np.transpose(variational_params.means_beta)}")
