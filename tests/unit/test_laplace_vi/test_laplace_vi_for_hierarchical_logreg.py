import numpy as np

from categorical_from_binary.data_generation.hierarchical_logreg import (
    generate_hierarchical_logistic_regression_dataset,
)
from categorical_from_binary.laplace_vi.hierarchical_logreg.inference import (
    GlobalHyperparams,
    compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi,
)


def test_compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi():
    """
    We test that the mean relative error on estimating the true global beta with the posterior mean
    for the global beta goes down as the dataset includes more samples.

    Note that, to make this a unit test (i.e. fast), we are:
        * using small values for the number of groups and number of features
        * just running a couple rounds of inference on each dataset, rather than running to convergence.

    This is just one possible test; there are many others we could write as well.
    """
    # initializations
    n_groups = 3
    n_features = 5
    beta_dim = n_features + 1
    beta_means_for_groups_init = np.zeros((n_groups, beta_dim))
    beta_mean_global_init = np.zeros(n_features + 1)
    beta_cov_global_init = np.eye(n_features + 1)
    # TODO: think more about better hyperparams
    global_hyperparams = GlobalHyperparams(
        nu=1, phi_0=np.eye(beta_dim), phi_1=np.eye(beta_dim)
    )

    n_samples = 10
    dataset_hierarchical = generate_hierarchical_logistic_regression_dataset(
        n_samples, n_features, n_groups, seed=123
    )
    variational_params = (
        compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi(
            dataset_hierarchical,
            beta_means_for_groups_init,
            beta_mean_global_init,
            beta_cov_global_init,
            global_hyperparams,
            n_iterations_for_vi=2,
            n_iterations_for_global_updates=2,
            display_output_vi=False,
        )
    )
    global_beta_posterior_mean = variational_params.global_.mean
    global_beta_true = dataset_hierarchical.beta_global
    mean_relative_error_with_10_samples = np.mean(
        abs(global_beta_posterior_mean - global_beta_true) / abs(global_beta_true)
    )

    n_samples = 50
    dataset_hierarchical = generate_hierarchical_logistic_regression_dataset(
        n_samples, n_features, n_groups, seed=123
    )
    variational_params = (
        compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi(
            dataset_hierarchical,
            beta_means_for_groups_init,
            beta_mean_global_init,
            beta_cov_global_init,
            global_hyperparams,
            n_iterations_for_vi=2,
            n_iterations_for_global_updates=2,
            display_output_vi=False,
        )
    )
    global_beta_posterior_mean = variational_params.global_.mean
    global_beta_true = dataset_hierarchical.beta_global
    mean_relative_error_with_50_samples = np.mean(
        abs(global_beta_posterior_mean - global_beta_true) / abs(global_beta_true)
    )

    n_samples = 250
    dataset_hierarchical = generate_hierarchical_logistic_regression_dataset(
        n_samples, n_features, n_groups, seed=123
    )
    variational_params = (
        compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi(
            dataset_hierarchical,
            beta_means_for_groups_init,
            beta_mean_global_init,
            beta_cov_global_init,
            global_hyperparams,
            n_iterations_for_vi=2,
            n_iterations_for_global_updates=2,
            display_output_vi=False,
        )
    )
    global_beta_posterior_mean = variational_params.global_.mean
    global_beta_true = dataset_hierarchical.beta_global
    mean_relative_error_with_250_samples = np.mean(
        abs(global_beta_posterior_mean - global_beta_true) / abs(global_beta_true)
    )

    assert (
        mean_relative_error_with_10_samples
        > mean_relative_error_with_50_samples
        > mean_relative_error_with_250_samples
    )
