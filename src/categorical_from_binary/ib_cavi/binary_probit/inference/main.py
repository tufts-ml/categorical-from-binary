"""
Variational inference for Bayesian probit regression. 

References:
    - Report in this repo
    - Consonni and Marin (2007)
"""

import warnings
from typing import Optional

import numpy as np

from categorical_from_binary.ib_cavi.binary_probit.elbo import compute_elbo
from categorical_from_binary.ib_cavi.binary_probit.inference.structs import (
    PriorType,
    VariationalBeta,
    VariationalParams,
    VariationalTaus,
    VariationalZs,
)
from categorical_from_binary.ib_cavi.binary_probit.inference.tau_helpers import (
    compute_expected_tau_reciprocal_matrix,
)
from categorical_from_binary.ib_cavi.trunc_norm import (
    compute_expected_value_normal_minus,
    compute_expected_value_normal_plus,
)
from categorical_from_binary.selection.hyperparameters import Hyperparameters
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def compute_variational_expectation_of_z(
    labels: NumpyArray1D,
    covariates: NumpyArray2D,
    beta: NumpyArray1D,
) -> NumpyArray1D:
    """
    The z's are continuous latent variables, one for each observation,  such that
        * y_i = 0 if z_i>0
        * y_i = 1 if z_i <=0.
    And the z_i's are related to the covariates and regression weights by
        z_i | beta ~ N (x_i' beta, 1)

    The function does not return the variational variance, because the variance is fixed.

    Returns:
        array with shape (n_observations, ), which is the variational expectation of
         z = (z_1, ..., z_N), where N is the number of observations.
    """
    n_obs = np.shape(covariates)[0]
    linear_predictors = covariates @ beta
    z_expected = np.zeros(n_obs)
    # TODO : This is unnecessarily slow for large N.  Find a way to vectorize this,
    # So that we can evaluate
    for i, (label, covariate_vec, linear_predictor) in enumerate(
        zip(labels, covariates, linear_predictors)
    ):
        if label == 0:
            z_expected[i] = compute_expected_value_normal_minus(linear_predictor)
        elif label == 1:
            z_expected[i] = compute_expected_value_normal_plus(linear_predictor)
        else:
            raise ValueError("The label should be either 0 or 1.")
    return z_expected


def update_taus(
    variational_beta: VariationalBeta, hyperparameters: Hyperparameters
) -> VariationalTaus:
    hp = hyperparameters
    qbeta = variational_beta
    return VariationalTaus(
        a=hp.lambda_ - 0.5,
        c=1 / (hp.gamma**2),
        d=qbeta.mean**2 + np.diag(qbeta.cov),
    )


def compute_variational_parameters_for_beta_under_normal_prior(
    X: NumpyArray2D, z_mean: NumpyArray1D
) -> VariationalBeta:
    """
    Computes variational parameters (variational mean, variational cov) for beta,
    assuming that our prior is N(0,I)
    """
    # TODO: Relax assumption that prior on beta is N(0,I)
    beta_dim = np.shape(X)[1]
    # TODO: compute this in advance.
    cov_beta = np.linalg.inv(np.eye(beta_dim) + X.T @ X)
    mu_beta = cov_beta @ X.T @ z_mean
    return VariationalBeta(mu_beta, cov_beta)


def compute_variational_parameters_for_beta_under_normal_gamma_prior(
    X: NumpyArray2D,
    z_mean: NumpyArray1D,
    variational_taus: VariationalTaus,
    hyperparameters: Hyperparameters,
) -> VariationalBeta:
    expected_tau_reciprocal_matrix = compute_expected_tau_reciprocal_matrix(
        variational_taus,
        hyperparameters,
    )
    # TODO: Precompute X.T @ X
    beta_cov = np.linalg.inv(expected_tau_reciprocal_matrix + X.T @ X)
    beta_mean = np.ndarray.flatten(beta_cov @ (X.T @ z_mean))
    return VariationalBeta(beta_mean, beta_cov)


def compute_probit_vi_with_normal_gamma_prior(
    labels: NumpyArray1D,
    covariates: NumpyArray2D,
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_elbo: float = -np.inf,
    variational_beta_init: Optional[VariationalBeta] = None,
    hyperparameters: Optional[Hyperparameters] = None,
    verbose: bool = True,
) -> VariationalParams:
    """
    Variational inference for Bayesian probit regression.
    Mostly follows Consonni and Marin (2007), with some Brown and Griffin (2010) at the top,
    but see my categorical_from_binary notes for a complete derivation.

    Arguments:
        hyperparameters:  Optional, does not need to be specified if prior_type is NORMAL.
            does need to be specified if prior_type is NORMAL_GAMMA
    """
    return _compute_probit_vi(
        labels,
        covariates,
        max_n_iterations,
        convergence_criterion_drop_in_elbo,
        variational_beta_init,
        prior_type=PriorType.NORMAL_GAMMA,
        hyperparameters=hyperparameters,
        verbose=verbose,
    )


def compute_probit_vi_with_normal_prior(
    labels: NumpyArray1D,
    covariates: NumpyArray2D,
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_elbo: float = -np.inf,
    variational_beta_init: Optional[VariationalBeta] = None,
    hyperparameters: Optional[Hyperparameters] = None,
    verbose: bool = True,
) -> VariationalParams:
    """
    Variational inference for Bayesian probit regression.
    Mostly follows Consonni and Marin (2007), but see my categorical_from_binary notes for a
    more complete derivation (inluding ELBO).  Currently assumes prior is beta~N(0,I).

    Arguments:
        hyperparameters:  Optional, does not need to be specified if prior_type is NORMAL.
            does need to be specified if prior_type is NORMAL_GAMMA
    """
    return _compute_probit_vi(
        labels,
        covariates,
        max_n_iterations,
        convergence_criterion_drop_in_elbo,
        variational_beta_init,
        prior_type=PriorType.NORMAL,
        hyperparameters=hyperparameters,
        verbose=verbose,
    )


def _compute_probit_vi(
    labels: NumpyArray1D,
    covariates: NumpyArray2D,
    max_n_iterations: float = np.inf,
    convergence_criterion_drop_in_elbo: float = -np.inf,
    variational_beta_init: Optional[VariationalBeta] = None,
    prior_type: PriorType = PriorType.NORMAL,
    hyperparameters: Optional[Hyperparameters] = None,
    verbose: bool = True,
) -> VariationalParams:
    """
    Variational inference for Bayesian probit regression.
    Mostly follows Consonni and Marin (2007).
    Currently assumes prior is beta~N(0,I).

    Arguments:
        hyperparameters:  Optional, does not need to be specified if prior_type is NORMAL.
            does need to be specified if prior_type is NORMAL_GAMMA
    """
    if prior_type == PriorType.NORMAL:
        warnings.warn(
            "We currently assume the prior on beta is N(0,I). Does that mean and variance make sense?"
        )

    if prior_type == PriorType.NORMAL_GAMMA and hyperparameters is None:
        raise ValueError(
            "If prior type is NORMAL_GAMMA, you must provide hyperparameters."
        )

    if max_n_iterations == np.inf and convergence_criterion_drop_in_elbo == -np.inf:
        raise ValueError(
            f"You must change max_n_iterations and/or convergence_criterion_drop_in_elbo "
            f"from the default value so that the algorithm knows when to stop"
        )

    if (
        prior_type == PriorType.NORMAL_GAMMA
        and convergence_criterion_drop_in_elbo != -np.inf
    ):
        raise ValueError(
            "We do not currently have a way to compute the ELBO under the normal-gamma prior."
        )

    _, n_covariates = np.shape(covariates)

    if variational_beta_init is None:
        warnings.warn(
            f"Initial value for variational expectation of beta is the vector of all 0's, "
            f"and for the covariance matrix is the identity matrix"
        )
        # TODO: Add support for warm-starting the inference with the MLE
        # variational initialization
        variational_beta_init = VariationalBeta(
            mean=np.zeros(n_covariates), cov=np.eye(n_covariates)
        )

    variational_taus = None
    variational_beta = variational_beta_init
    n_iterations_so_far = 0
    previous_elbo, elbo, drop_in_elbo = (
        -np.inf,
        -np.inf,
        np.inf,
    )
    print(
        f"Max # iterations: {max_n_iterations}.  Convergence criterion (drop in ELBO): {convergence_criterion_drop_in_elbo}"
    )

    while (
        n_iterations_so_far <= max_n_iterations
        and drop_in_elbo >= convergence_criterion_drop_in_elbo
    ):
        z_mean = compute_variational_expectation_of_z(
            labels, covariates, variational_beta.mean
        )

        if prior_type == PriorType.NORMAL:
            variational_beta = (
                compute_variational_parameters_for_beta_under_normal_prior(
                    covariates, z_mean
                )
            )
        elif prior_type == PriorType.NORMAL_GAMMA:
            variational_taus = update_taus(variational_beta, hyperparameters)
            variational_beta = (
                compute_variational_parameters_for_beta_under_normal_gamma_prior(
                    covariates, z_mean, variational_taus, hyperparameters
                )
            )
        else:
            raise ValueError(f"Prior type {prior_type} is unknown.")
        if verbose:
            print(f"Iteration: {n_iterations_so_far}.")
        if convergence_criterion_drop_in_elbo != -np.inf:
            elbo = compute_elbo(
                variational_beta.mean,
                variational_beta.cov,
                z_mean,
                covariates,
                labels,
                verbose,
            )
            drop_in_elbo = elbo - previous_elbo
            previous_elbo = elbo
        if verbose:
            print(f"Iteration: {n_iterations_so_far}.  ELBO: {elbo}.", end="\r")
        n_iterations_so_far += 1

    z_natural_params = covariates @ variational_beta.mean
    variational_zs = VariationalZs(z_natural_params)
    return VariationalParams(variational_beta, variational_zs, variational_taus)
