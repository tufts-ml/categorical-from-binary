import numpy as np

from categorical_from_binary.data_generation.hierarchical_logreg import (
    generate_hierarchical_logistic_regression_dataset,
)
from categorical_from_binary.laplace_vi.hierarchical_logreg.inference import (
    GlobalHyperparams,
    compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi,
)
from categorical_from_binary.laplace_vi.hierarchical_logreg.sanity import (
    sanity_check_group_betas,
    sanity_check_hierarchical_model,
)


def test_compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi_for_small_group_sizes():
    print(
        f"\n\nNow testing Laplace Variational Inference with Hierarchical Bayesian Logistic Regression \n"
        f"when group sizes are small. In particular, does the hierarchical model have less error than independent models?"
    )
    n_samples, n_features, n_groups = 100, 5, 10
    dataset_hierarchical = generate_hierarchical_logistic_regression_dataset(
        n_samples, n_features, n_groups, seed=123
    )

    beta_dim = n_features + 1
    beta_means_for_groups_init = np.zeros((n_groups, beta_dim))
    beta_mean_global_init = np.zeros(n_features + 1)
    beta_cov_global_init = np.eye(n_features + 1)
    # TODO: think more about better hyperparams
    global_hyperparams = GlobalHyperparams(
        nu=1, phi_0=np.eye(beta_dim), phi_1=np.eye(beta_dim)
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
        )
    )

    group_beta_estimation_errors = sanity_check_hierarchical_model(
        variational_params, dataset_hierarchical
    )
    assert np.mean(group_beta_estimation_errors.hierarchical) < np.mean(
        group_beta_estimation_errors.independent
    )


def test_compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi_for_large_group_sizes_and_lots_of_inference():
    print(
        f"\n\nNow testing Laplace Variational Inference with Hierarchical Bayesian Logistic Regression \n"
        f"wehn group sizes are large and we do lots of inference.  In particular, is the posterior means (for global and group-specific betas)"
        f"close to the true generating regression weights for n large?"
    )
    n_samples, n_features, n_groups = 1000, 5, 10
    dataset_hierarchical = generate_hierarchical_logistic_regression_dataset(
        n_samples, n_features, n_groups, seed=123
    )

    beta_dim = n_features + 1
    beta_means_for_groups_init = np.zeros((n_groups, beta_dim))
    beta_mean_global_init = np.zeros(n_features + 1)
    beta_cov_global_init = np.eye(n_features + 1)
    # TODO: think more about better hyperparams
    global_hyperparams = GlobalHyperparams(
        nu=1, phi_0=np.eye(beta_dim), phi_1=np.eye(beta_dim)
    )

    variational_params = (
        compute_variational_posterior_for_hierarchical_bayesian_logreg_using_laplace_vi(
            dataset_hierarchical,
            beta_means_for_groups_init,
            beta_mean_global_init,
            beta_cov_global_init,
            global_hyperparams,
            n_iterations_for_vi=10,
            n_iterations_for_global_updates=5,
        )
    )

    group_beta_estimation_data = sanity_check_group_betas(
        variational_params, dataset_hierarchical
    )

    # TODO: also add sanity check on the global betas (which we should have even less error in)

    TOLERANCE_FOR_MEAN_RELATIVE_ERROR_IN_ESTIMATING_GROUP_BETA_ENTRIES = 0.25
    TOLERANCE_FOR_MEDIAN_RELATIVE_ERROR_IN_ESTIMATING_GROUP_BETA_ENTRIES = 0.15
    relative_errors = np.zeros((n_groups, n_features + 1))
    for group, (posterior_mean, true_beta) in enumerate(
        zip(group_beta_estimation_data.posterior_mean, group_beta_estimation_data.true)
    ):
        relative_errors[group] = abs(posterior_mean - true_beta) / abs(true_beta)

    assert (
        np.mean(relative_errors)
        < TOLERANCE_FOR_MEAN_RELATIVE_ERROR_IN_ESTIMATING_GROUP_BETA_ENTRIES
    )
    assert (
        np.median(relative_errors)
        < TOLERANCE_FOR_MEDIAN_RELATIVE_ERROR_IN_ESTIMATING_GROUP_BETA_ENTRIES
    )
