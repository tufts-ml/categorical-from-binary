from typing import NamedTuple

import numpy as np

from categorical_from_binary.data_generation.bayes_binary_reg import (
    BinaryRegressionDataset,
)
from categorical_from_binary.data_generation.util import (
    prepend_features_with_column_of_all_ones_for_intercept,
)
from categorical_from_binary.polya_gamma.binary_logreg_vi.util import (
    compute_log_abs_det,
    compute_matrix_inverse,
)
from categorical_from_binary.polya_gamma.polya_gamma import (
    compute_polya_gamma_expectation,
)
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


class VariationalParameters(NamedTuple):
    mean_beta: NumpyArray1D
    cov_beta: NumpyArray2D


class PriorParameters(NamedTuple):
    mean_beta: NumpyArray1D
    cov_beta: NumpyArray2D


def compute_expected_value_of_c_parameter_for_polya_gamma(
    features: NumpyArray2D,
    variational_mean_beta: NumpyArray1D,
    variational_cov_beta: NumpyArray2D,
) -> float:

    expected_linear_predictors = features @ variational_mean_beta
    quadratic_forms = np.array(
        [np.transpose(x_i) @ variational_cov_beta @ x_i for x_i in features]
    )
    expected_natural_parameters = quadratic_forms + expected_linear_predictors**2
    return np.sqrt(expected_natural_parameters)


###
# ELBO Computation
###


def compute_kl_divergence_from_prior_to_variational_beta(
    variational_mean_beta: NumpyArray1D,
    variational_cov_beta: NumpyArray2D,
    prior_mean_beta: NumpyArray1D,
    prior_cov_beta: NumpyArray2D,
) -> float:

    # TODO: Could really write this as KL divergence between from one MVN to another,
    # then call that with this function,  using the appropriate inputs

    # aliases
    (mu_0, inv_Sigma_0) = prior_mean_beta, compute_matrix_inverse(prior_cov_beta)
    (qmu, qSigma) = variational_mean_beta, variational_cov_beta

    LD_inv_Sigma_0 = compute_log_abs_det(inv_Sigma_0)
    LD_qSigma = compute_log_abs_det(qSigma)

    # number of dimensions
    d = len(variational_mean_beta)
    result_times_two = (
        np.trace(inv_Sigma_0 @ qSigma)
        + np.transpose(qmu - mu_0) @ inv_Sigma_0 @ (qmu - mu_0)
        - d
        - LD_inv_Sigma_0
        - LD_qSigma
    )

    return 0.5 * result_times_two


def compute_expected_log_likelihood_plus_omega_entropy(
    features: NumpyArray2D,
    labels: NumpyArray1D,
    variational_mean_beta: NumpyArray1D,
    variational_cov_beta: NumpyArray2D,
) -> float:
    expected_c_parameters = compute_expected_value_of_c_parameter_for_polya_gamma(
        features,
        variational_mean_beta,
        variational_cov_beta,
    )
    return np.sum(
        features @ variational_mean_beta * (labels - 0.5)
        - 0.5 * expected_c_parameters
        - np.log(1 + np.exp(-expected_c_parameters))
    )


def compute_elbo(
    features: NumpyArray2D,
    labels: NumpyArray1D,
    beta_mean: NumpyArray1D,
    beta_cov: NumpyArray2D,
    prior_beta_mean: NumpyArray1D,
    prior_beta_cov: NumpyArray2D,
) -> float:
    """"""
    # TODO: Set up an alternate function where I precompute the inverse
    # prior_precision_beta = np.linalg.inv(prior_cov_beta)

    kl_divergence_of_beta_to_prior = (
        compute_kl_divergence_from_prior_to_variational_beta(
            beta_mean,
            beta_cov,
            prior_beta_mean,
            prior_beta_cov,
        )
    )

    expected_log_likelihood_plus_omega_entropy = (
        compute_expected_log_likelihood_plus_omega_entropy(
            features,
            labels,
            beta_mean,
            beta_cov,
        )
    )
    return -kl_divergence_of_beta_to_prior + expected_log_likelihood_plus_omega_entropy


###
# Main inference routine
###


def run_polya_gamma_variational_inference_for_bayesian_logistic_regression(
    dataset: BinaryRegressionDataset,
    prior_params: PriorParameters,
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_elbo: float = -np.inf,
    verbose: bool = False,
    prepend_features_with_column_of_all_ones: bool = True,
) -> VariationalParameters:
    """
    Use variational inference with polya gamma augementation to approximate the posterior mean and covariance
    on the regression weights of a Bayesian logistic regression

    Arguments:
        prior_mean:  the prior expectation for the regression weights
        prior_cov:  the prior covariance for the regression weights
    """
    if max_n_iterations == np.inf and convergence_criterion_drop_in_elbo == -np.inf:
        raise ValueError(
            f"You must change max_n_iterations and/or convergence_criterion_drop_in_elbo "
            f"from the default value so that the algorithm knows when to stop"
        )

    # initialization
    prior_mean, prior_cov = prior_params
    prior_precision = np.linalg.inv(prior_cov)
    if prepend_features_with_column_of_all_ones:
        features = prepend_features_with_column_of_all_ones_for_intercept(
            dataset.features
        )
    else:
        features = dataset.features
    features_transposed = np.transpose(features)
    kappa = dataset.labels - 0.5

    # initialization for top-level variational parameters
    variational_mean_beta = prior_mean
    variational_cov_beta = prior_cov

    n_iterations_so_far = 0
    previous_elbo, drop_in_elbo = -np.inf, np.inf
    print(
        f"Max # iterations: {max_n_iterations}.  Convergence criterion (drop in ELBO): {convergence_criterion_drop_in_elbo}"
    ) if verbose else None
    print(f"\nTrue beta: {dataset.beta}\n") if verbose else None
    while (
        n_iterations_so_far < max_n_iterations
        and drop_in_elbo >= convergence_criterion_drop_in_elbo
    ):
        expected_c_parameters = compute_expected_value_of_c_parameter_for_polya_gamma(
            features,
            variational_mean_beta,
            variational_cov_beta,
        )
        polya_gamma_expectations = np.array(
            [compute_polya_gamma_expectation(1, c) for c in expected_c_parameters]
        )

        # TODO: the X'X might be very intensive with many samples and few covariates.  could perhaps use random projection.
        variational_cov_beta = np.linalg.inv(
            features_transposed @ (polya_gamma_expectations[:, None] * features)
            + prior_precision
        )
        variational_mean_beta = variational_cov_beta @ (
            features_transposed @ kappa + prior_precision @ prior_mean
        )
        variational_params = VariationalParameters(
            variational_mean_beta, variational_cov_beta
        )

        elbo = compute_elbo(
            features,
            dataset.labels,
            variational_params.mean_beta,
            variational_params.cov_beta,
            prior_params.mean_beta,
            prior_params.cov_beta,
        )
        drop_in_elbo = elbo - previous_elbo
        previous_elbo = elbo
        n_iterations_so_far += 1
        print(
            f"At iteration {n_iterations_so_far}, the ELBO is {elbo}, and the variational mean is: {variational_params.mean_beta}"
        ) if verbose else None
        # print(f"Drop in elbo: {drop_in_elbo}")
    return variational_params
