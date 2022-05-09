"""
Implement Bayesian logistic regression using methods for nonconjugate VI,
as per Wang and Blei (2013).

Currently will implement Laplace VI.

Note that for non-hierarchical models, such as what is studied here, Laplace
VI gives the same results as does a Laplace approximation (Bishop 2006)
"""

from typing import Callable, Tuple

import numpy as np
import scipy
from scipy.optimize import minimize

from categorical_from_binary.data_generation.bayes_binary_reg import BinaryRegressionDataset
from categorical_from_binary.kl import sigmoid


###
# Setup Loss Fucntion
###


def make_loss_function_regularizer(
    prior_mean: np.ndarray, prior_precision: np.ndarray
) -> Callable:
    return (
        lambda beta: -0.5
        * np.transpose(beta - prior_mean)
        @ prior_precision
        @ (beta - prior_mean)
    )


def loss_function_likelihood_core(
    beta: np.ndarray, features: np.ndarray, labels: np.ndarray
) -> float:
    """
    This is the sum of the log model probabilities with the given parameter

    Arguments:
        beta:  vector of regression weights,  has shape (p+1, ) where p is the number of covariates
            The 0-th position gives beta_0, i.e. the intercept

    """
    result = 0
    for (i, label) in enumerate(labels):
        unsigned_linear_predictor = beta[0] + np.dot(beta[1:], features[i, :])
        sign_of_linear_predictor = (label == 0) * -1
        linear_predictor = unsigned_linear_predictor * sign_of_linear_predictor
        prob = sigmoid(linear_predictor)
        result += np.log(prob)
    return result
    # for unit testing...
    # param=Param(log_reg_dataset.beta_0,  log_reg_dataset.beta)


def make_calc_loss_function(prior_mean: np.ndarray, prior_cov: np.ndarray) -> Callable:
    prior_precision = scipy.linalg.inv(prior_cov)
    loss_function_regularizer = make_loss_function_regularizer(
        prior_mean, prior_precision
    )
    return lambda beta, features, labels: -(
        loss_function_likelihood_core(beta, features, labels)
        + loss_function_regularizer(beta)
    )


###
# Jacobian computation
###


def make_likelihood_contribution_to_jacobian(
    beta: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """

    Calculate the likelihood's contribution to the Jacobian for the Laplace update to the parameter beta
    for a Bayesian Logistic Regression model

    Arguments:
        beta:  vector of regression weights,  has shape (p+1, ) where p is the number of covariates
            The 0-th position gives beta_0, i.e. the intercept

    Returns:
        np.ndarray of shape (p+1, )
    """
    jacobian_without_regularization_term = np.zeros_like(beta)
    for (i, label) in enumerate(labels):
        linear_predictor = beta[0] + np.dot(beta[1:], features[i, :])
        jacobian_without_regularization_term[0] += 1.0 * (
            label - sigmoid(linear_predictor)
        )
        jacobian_without_regularization_term[1:] += features[i, :] * (
            label - sigmoid(linear_predictor)
        )
    return jacobian_without_regularization_term


def make_prior_contribution_to_jacobian_function(
    prior_mean: np.ndarray, prior_cov: np.ndarray
) -> Callable:
    prior_precision = scipy.linalg.inv(prior_cov)
    return lambda beta: -prior_precision @ (beta - prior_mean)


def make_function_to_calc_jacobian_for_laplace_step_in_vi_for_bayesian_log_reg(
    prior_mean: np.ndarray, prior_cov: np.ndarray
) -> Callable:
    prior_contribution_to_jacobian_function = (
        make_prior_contribution_to_jacobian_function(prior_mean, prior_cov)
    )
    return lambda beta, features, labels: -(
        make_likelihood_contribution_to_jacobian(beta, features, labels)
        + prior_contribution_to_jacobian_function(beta)
    )


###
# Hessian computation
###


def make_likelihood_contribution_to_hessian(
    beta: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """

    Calculate the likelihood's contribution to the Hessian for the Laplace update to the parameter beta
    for a Bayesian Logistic Regression model

    Arguments:
        beta:  vector of regression weights,  has shape (p+1, ) where p is the number of covariates
            The 0-th position gives beta_0, i.e. the intercept

    Returns:
        np.ndarray of shape (p+1, )
    """
    hessian_without_regularization_term = np.zeros((len(beta), len(beta)))
    for (i, label) in enumerate(labels):
        linear_predictor = beta[0] + np.dot(beta[1:], features[i, :])
        weight = -sigmoid(linear_predictor) * sigmoid(-linear_predictor)
        features_with_one = np.concatenate(([1], features[i, :]))
        hessian_without_regularization_term += weight * np.outer(
            features_with_one, features_with_one
        )
    return hessian_without_regularization_term


def make_prior_contribution_to_hessian_function(prior_cov: np.ndarray) -> np.ndarray:
    prior_precision = scipy.linalg.inv(prior_cov)
    return -prior_precision


def make_function_to_calc_hessian_for_laplace_step_in_vi_for_bayesian_log_reg(
    prior_mean: np.ndarray, prior_cov: np.ndarray
) -> Callable:
    prior_contribution_to_hessian_function = (
        make_prior_contribution_to_hessian_function(prior_cov)
    )
    return lambda beta, features, labels: -(
        make_likelihood_contribution_to_hessian(beta, features, labels)
        + prior_contribution_to_hessian_function
    )


###
# Get fitted beta
###


def optimize_beta_for_bayesian_logreg_using_laplace_vi(
    beta_init: np.array,
    dataset: BinaryRegressionDataset,
    prior_mean: np.array,
    prior_cov: np.array,
    display_output: bool = True,
) -> np.ndarray:
    """
    Find the value of beta that optimizes the exponent of the exponential expression giving the variational
    update to beta in variational inference.  See Equation (12) in Wang and Blei, 2013, JMLR
    """
    calc_loss = make_calc_loss_function(prior_mean, prior_cov)
    calc_log_reg_jacobian = (
        make_function_to_calc_jacobian_for_laplace_step_in_vi_for_bayesian_log_reg(
            prior_mean, prior_cov
        )
    )
    calc_log_reg_hessian = (
        make_function_to_calc_hessian_for_laplace_step_in_vi_for_bayesian_log_reg(
            prior_mean, prior_cov
        )
    )
    result = minimize(
        calc_loss,
        beta_init,
        args=(dataset.features, dataset.labels),
        method="Newton-CG",
        jac=calc_log_reg_jacobian,
        hess=calc_log_reg_hessian,
        options={"xtol": 1e-8, "disp": display_output},
    )
    return result.x


def get_beta_covariance(
    beta_mean: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    dataset: BinaryRegressionDataset,
) -> np.ndarray:
    calc_log_reg_hessian = (
        make_function_to_calc_hessian_for_laplace_step_in_vi_for_bayesian_log_reg(
            prior_mean, prior_cov
        )
    )
    beta_cov = scipy.linalg.inv(
        calc_log_reg_hessian(beta_mean, dataset.features, dataset.labels)
    )
    return beta_cov


def get_variational_params_for_regression_weights_of_bayesian_logreg_using_laplace_vi(
    beta_init: np.array,
    dataset: BinaryRegressionDataset,
    prior_mean: np.array,
    prior_cov: np.array,
    display_output: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the mean and covariance of the variational normal distribution governing the regression weights, beta,
    of a Bayesian logistic regression where we handle the nonconjugacy of beta by using Laplace variational inference.
    See Equation (13) in Wang and Blei, 2013, JMLR
    """
    beta_mean = optimize_beta_for_bayesian_logreg_using_laplace_vi(
        beta_init,
        dataset,
        prior_mean,
        prior_cov,
        display_output,
    )
    beta_cov = get_beta_covariance(beta_mean, prior_mean, prior_cov, dataset)
    return beta_mean, beta_cov
