import numpy as np
import scipy

from categorical_from_binary.data_generation.bayes_binary_reg import (
    generate_logistic_regression_dataset,
)
from categorical_from_binary.laplace_vi.bayes_logreg.inference import (
    get_variational_params_for_regression_weights_of_bayesian_logreg_using_laplace_vi,
    make_calc_loss_function,
)


def test_loss_calculation_for_laplace_update_for_bayesian_logistic_regression(
    logistic_regression_dataset,
):
    dataset = logistic_regression_dataset
    beta_dim = len(dataset.beta)
    prior_mean = np.zeros(beta_dim)
    prior_cov = np.eye(beta_dim)
    calc_loss = make_calc_loss_function(prior_mean, prior_cov)

    beta_init = np.zeros(beta_dim)
    loss_init = calc_loss(beta_init, dataset.features, dataset.labels)

    beta_true = dataset.beta
    loss_true = calc_loss(beta_true, dataset.features, dataset.labels)

    assert loss_true < loss_init


def test_get_variational_params_for_regression_weights_of_bayesian_logreg_using_laplace_vi():
    """
    The posterior covariance matrix on the regression weights, beta, should have larger norm when we process on fewer samples.
    """
    n_features = 5
    beta_init = np.zeros(n_features + 1)
    prior_mean = np.zeros(n_features + 1)
    prior_cov = np.eye(n_features + 1)

    n_samples = 10
    dataset = generate_logistic_regression_dataset(n_samples, n_features, seed=1)
    (
        mean_beta,
        cov_beta,
    ) = get_variational_params_for_regression_weights_of_bayesian_logreg_using_laplace_vi(
        beta_init,
        dataset,
        prior_mean,
        prior_cov,
        display_output=False,
    )
    norm_of_cov_matrix_with_10_samples = scipy.linalg.norm(cov_beta)

    n_samples = 100
    dataset = generate_logistic_regression_dataset(n_samples, n_features, seed=1)
    (
        mean_beta,
        cov_beta,
    ) = get_variational_params_for_regression_weights_of_bayesian_logreg_using_laplace_vi(
        beta_init,
        dataset,
        prior_mean,
        prior_cov,
        display_output=False,
    )
    norm_of_cov_matrix_with_100_samples = scipy.linalg.norm(cov_beta)

    n_samples = 1000
    dataset = generate_logistic_regression_dataset(n_samples, n_features, seed=1)
    (
        mean_beta,
        cov_beta,
    ) = get_variational_params_for_regression_weights_of_bayesian_logreg_using_laplace_vi(
        beta_init,
        dataset,
        prior_mean,
        prior_cov,
        display_output=False,
    )
    norm_of_cov_matrix_with_1000_samples = scipy.linalg.norm(cov_beta)

    assert (
        norm_of_cov_matrix_with_10_samples
        > norm_of_cov_matrix_with_100_samples
        > norm_of_cov_matrix_with_1000_samples
    )
