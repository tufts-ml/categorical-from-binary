import numpy as np

from categorical_from_binary.data_generation.hierarchical_logreg import (
    generate_hierarchical_logistic_regression_dataset,
)
from categorical_from_binary.laplace_vi.hierarchical_logreg.inference import (
    GlobalHyperparams,
    compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi,
)
from categorical_from_binary.laplace_vi.hierarchical_logreg.sanity import (
    sanity_check_global_betas,
    sanity_check_group_betas,
    sanity_check_hierarchical_model,
)


###
# Generate Data
####

n_samples, n_features, n_groups = 100, 5, 10
dataset_hierarchical = generate_hierarchical_logistic_regression_dataset(
    n_samples, n_features, n_groups, seed=123
)

###
# Variational Inference
####

### Initializations
beta_dim = n_features + 1
beta_means_for_groups_init = np.zeros((n_groups, beta_dim))
beta_mean_global_init = np.zeros(beta_dim)
beta_cov_global_init = np.eye(beta_dim)
# TODO: think more about better hyperparams
global_hyperparams = GlobalHyperparams(
    nu=1, phi_0=np.eye(beta_dim), phi_1=np.eye(beta_dim)
)

### Optimization
variational_params = (
    compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi(
        dataset_hierarchical,
        beta_means_for_groups_init,
        beta_mean_global_init,
        beta_cov_global_init,
        global_hyperparams,
        n_iterations_for_vi=2,
        n_iterations_for_global_updates=2,
    )
)

###
# Sanity Checks
####
sanity_check_global_betas(variational_params, dataset_hierarchical)
sanity_check_group_betas(variational_params, dataset_hierarchical)
sanity_check_hierarchical_model(variational_params, dataset_hierarchical)
