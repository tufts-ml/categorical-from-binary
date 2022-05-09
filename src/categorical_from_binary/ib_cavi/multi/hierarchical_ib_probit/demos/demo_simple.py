"""
Compare the (fitted) choice probabilities for the hierarchical vs.
a bunch of separate models in standard, baseline case. 
"""

import numpy as np


np.set_printoptions(suppress=True, precision=3)

from categorical_from_binary.data_generation.bayes_multiclass_reg import Link
from categorical_from_binary.data_generation.hierarchical_multiclass_reg import (
    generate_hierarchical_multiclass_regression_dataset,
)
from categorical_from_binary.evaluate.hierarchical_multiclass import (
    compute_mean_choice_probs_for_hierarchical_multiclass_regression,
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


###
#  Generate data
###

n_samples = 50
n_features_exogenous = 3
n_categories = 3
n_groups = 5
s2_beta_expected = 1.0

data = generate_hierarchical_multiclass_regression_dataset(
    n_samples,
    n_features_exogenous,
    n_categories,
    n_groups,
    s2_beta_expected=s2_beta_expected,
)


###
#  Run the hierarchical model (and collect choice probabilities)
###

convergence_criterion_drop_in_elbo_for_hierarchical_multiclass_model = 0.1

variational_params = run_hierarchical_multiclass_probit_vi(
    data,
    convergence_criterion_drop_in_elbo=convergence_criterion_drop_in_elbo_for_hierarchical_multiclass_model,
)

mean_choice_probs_for_hierarchical_model = (
    compute_mean_choice_probs_for_hierarchical_multiclass_regression(
        variational_params,
        data,
    )
)

###
#  Run separate flat models (and collect choice probabilities)
###

mean_likelihoods_for_separate_models = [None] * n_groups
convergence_criterion_drop_in_elbo_for_separate_multiclass_models = 0.1

for j in range(n_groups):
    dataset = data.datasets[j]
    results = compute_multiclass_probit_vi_with_normal_prior(
        dataset.labels,
        dataset.features,
        convergence_criterion_drop_in_elbo=convergence_criterion_drop_in_elbo_for_separate_multiclass_models,
    )
    beta_mean, beta_cov = (
        results.variational_params.beta.mean,
        results.variational_params.beta.cov,
    )
    mean_likelihoods_for_separate_models[
        j
    ] = compute_mean_likelihood_for_multiclass_regression(
        dataset.features,
        dataset.labels,
        beta_mean,
        link_for_category_probabilities=Link.CBM_PROBIT,
    )

###
# Compare
###

J, M, K, Ns = data_dimensions_from_hierarchical_dataset(data)
print(
    f"\nNum groups: {J}"
    f"\nMean number of observations per group: {np.mean(Ns):.04}"
    f"\nNum categories: {K}"
    f"\nNum designed features: {M}"
    f"\nScalar for isotropic variance (s2) controlling regression weight variance across groups: {s2_beta_expected}"
)

print(f"\nFor {n_groups} separate models, the mean choice probs were: ")
print([f"{p:.03}" for p in mean_likelihoods_for_separate_models])

print(f"\nFor the hierarchical model, the mean choice probs were: ")
print([f"{p:.03}" for p in mean_choice_probs_for_hierarchical_model])
