from pprint import pprint

import numpy as np

from categorical_from_binary.data_generation.hierarchical_multiclass_reg import (
    generate_hierarchical_multiclass_regression_dataset,
)


###
#  A dataset with strong correlations between labels over time.
###

print(
    f"\nA hierarchical categorical dataset where there are strong correlations between "
    f"labels over time."
)


n_samples = 50
n_features_exogenous = 1
n_categories = 3
n_groups = 10

beta_transition_expected = np.array(
    [
        [2.0, 0.2, 0.2],
        [0.2, 2.0, 0.2],
        [0.2, 0.2, 2.0],
    ]
)

# Remark:
# 1) tendency to go INTO a category can be influenced by changing the
# column sum of the k-th category.
# 2) tendency to go from a category to another category can be influenced
# by changing the relative proportions in a row (although note that the presence
# of external covariates and/or an intercept term must also be considered)

# 1.8*np.eye(n_categories) + 0.2*np.ones((n_categories, n_categories))

hd = generate_hierarchical_multiclass_regression_dataset(
    n_samples,
    n_features_exogenous,
    n_categories,
    n_groups,
    is_autoregressive=True,
    include_intercept=False,
    s2_beta_expected=0.1,
    s2_beta=0.1,
    beta_transition_expected=beta_transition_expected,
)

print("Example labels from one sequence in the collection:")
pprint(hd.datasets[0].labels[:20])


###
#  A dataset where the first category is pretty rare
###

print("\nA hierarchical categorical dataset where the first category is pretty rare")

n_samples = 50
n_features_exogenous = 1
n_categories = 3
n_groups = 10

beta_0_expected = np.array([-1.0, 0.0, 0.0])

hd = generate_hierarchical_multiclass_regression_dataset(
    n_samples,
    n_features_exogenous,
    n_categories,
    n_groups,
    is_autoregressive=False,
    include_intercept=True,
    s2_beta_expected=0.1,
    s2_beta=0.1,
    beta_0_expected=beta_0_expected,
)

category_counts_per_sequence = [sum(dataset.labels, 0) for dataset in hd.datasets]
print("Category counts per sequence")
pprint(category_counts_per_sequence)


"""
For most individual time series, will have very poor learning about
the first category,  even if there are predictive covariates. 

In [27]: [sum(dataset.labels,0) for dataset in hd.datasets]
Out[27]: 
[array([ 1, 21, 28]),
 array([ 1, 43,  6]),
 array([ 0, 22, 28]),
 array([18, 20, 12]),
 array([ 6, 23, 21]),
 array([ 4, 28, 18]),
 array([ 6, 24, 20]),
 array([ 2, 31, 17]),
 array([ 5, 30, 15]),
 array([15, 10, 25])]
 """


###
#  A dataset where the first category is pretty rare AND there is a predictive covariate.
###

print(
    f"\nA hierarchical categorical dataset where the first category is pretty rare "
    f"BUT there is a highly predictive covariate."
)


n_samples = 50
n_features_exogenous = 1
n_categories = 3
n_groups = 10


beta_0_expected = np.array([-1.0, 0.0, 0.0])
beta_exogenous_expected = np.array([[2.0, 0.0, 0.0]])

hd = generate_hierarchical_multiclass_regression_dataset(
    n_samples,
    n_features_exogenous,
    n_categories,
    n_groups,
    is_autoregressive=False,
    include_intercept=True,
    s2_beta_expected=0.1,
    s2_beta=0.5,
    beta_0_expected=beta_0_expected,
    beta_exogenous_expected=beta_exogenous_expected,
    indices_of_exogenous_features_that_are_binary=[0],
    success_probs_for_exogenous_features_that_are_binary=[0.05],
)


category_counts_per_sequence = [sum(dataset.labels, 0) for dataset in hd.datasets]
print("Category counts per sequence")
pprint(category_counts_per_sequence)


# TODO: could even have the rare, predictive covariate masked by another
# more common and more moderately difficult covariate - a harder learning problem.
