"""
We report the mean rank of the predicted label.

We also investigate if the inference algorithm is just always picking the label with the highest frequencies, 
or if it can also do a good job of handling minority categories.

Finally, we compare against how softmax does, with parameters estimated by automatic differentiation.
"""

import numpy as np


np.set_printoptions(precision=3, suppress=True)

from categorical_from_binary.autodiff.jax_helpers import optimize_beta_for_softmax_model
from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    generate_multiclass_regression_dataset,
)
from categorical_from_binary.ib_cavi.multi.ib_probit.inference.main import (
    compute_multiclass_probit_vi_with_normal_prior,
)
from categorical_from_binary.metrics import print_performance_report


n_categories = 5
n_sparse_categories = 0
n_features = 20
n_sparse_features = 0
n_samples = 5000
n_train_samples = 4000
include_intercept = True
link = Link.MULTI_LOGIT
seed = 1
dataset = generate_multiclass_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_categories=n_categories,
    n_categories_where_all_beta_coefficients_are_sparse=n_sparse_categories,
    n_sparse_features=n_sparse_features,
    beta_0=None,
    link=link,
    seed=seed,
    include_intercept=include_intercept,
)
np.set_printoptions(edgeitems=4)


# Prep training / test split
covariates_train = dataset.features[:n_train_samples]
labels_train = dataset.labels[:n_train_samples]
covariates_test = dataset.features[n_train_samples:]
labels_test = dataset.labels[n_train_samples:]


results = compute_multiclass_probit_vi_with_normal_prior(
    labels_train,
    covariates_train,
    labels_test=labels_test,
    covariates_test=covariates_test,
    variational_params_init=None,
    max_n_iterations=50,
    # convergence_criterion_drop_in_mean_elbo=0.01,
)


performance_over_time_CAVI = results.performance_over_time

print(f"\n\nCAVI holdout performance over time: \n {performance_over_time_CAVI }")

beta_mean = results.variational_params.beta.mean


print("Now using AD to optimize regression weights for the softmax likelihood")
beta_init = np.zeros((n_features + 1, n_categories))
beta_star_softmax_MLE = optimize_beta_for_softmax_model(
    beta_init,
    dataset.features,
    dataset.labels,
    verbose=False,
)


print("\n ----  Results with CBM probit model ---- ")
print_performance_report(
    covariates_test,
    labels_test,
    labels_train,
    beta_mean,
    link=Link.CBM_PROBIT,
)

print("\n ---- Results with softmax model ---- ")
print_performance_report(
    covariates_test,
    labels_test,
    labels_train,
    beta_star_softmax_MLE,
    link=Link.MULTI_LOGIT,
)
