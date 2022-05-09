"""
This module provides a Gibbs sampler for sampling from a multinomial logit model
with polya gamma augmentation.  There is no closed-form CAVI for this, as presented
in both the categorical_from_binary report and the appendix of the ICML paper, but there
is a straightforward Gibbs sampler.
"""
import sys

import numpy as np


np.set_printoptions(precision=3, suppress=True)
from dataclasses import dataclass
from typing import Optional

import pypolyagamma
import scipy
from pypolyagamma import PyPolyaGamma
from scipy.sparse import isspmatrix
from scipy.stats import multivariate_normal as mvn

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    compute_linear_predictors_preventing_downstream_overflow,
)
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D, NumpyArray3D


###
# Helper functions
###


def compute_log_sum_exponentiated_utilities_for_non_self_categories(
    features: NumpyArray2D,
    beta: NumpyArray2D,
):
    """
    Arguments:
        features: an np.array of shape (N,M)
            where N is the number of samples and M is the number of covariates
        beta: regression weights with shape (M,K),
            where M is the number of covariates and K is the number of categories

    Returns:
        an np.array with shape (N,K).
        The (i,k)-th element is
            C_ik = log sum_{j!=k} exp (eta_{ik}),
        a quantity used in the multinomial logit
        with polya gamma augmentation, where one works with the conditional likelihoods that have the form of
        logistic regression on the class indicators.

    """
    N, K = np.shape(features)[0], np.shape(beta)[1]
    eta = compute_linear_predictors_preventing_downstream_overflow(features, beta)
    exponentiated_etas = np.exp(eta)
    result = np.zeros((N, K))
    for k in range(K):
        mask = np.ones(K)
        mask[k] = 0
        # TODO: the below returns a copy; will be memory intensive.  better to work with a view.
        exponentiated_etas_without_kth_column = np.compress(
            mask, exponentiated_etas, axis=1
        )
        result[:, k] = np.log(np.sum(exponentiated_etas_without_kth_column, 1))
    return result


###
# Prior stuff
###


@dataclass
class PriorInfo:
    """
    Prior information represented in a way that is used for the Gibbs sampler.

    Attributes:
        precision:  Sigma_0^{-1}, where Sigma_0 is the prior variance
        precision_weighted_mean: Sigma_0^{-1} @ mu_0,
            where Sigma_0 is the prior variance and mu_0 is the prior mean
    """

    precision: NumpyArray2D
    precision_weighted_mean: NumpyArray1D


def prior_info_from_prior_params(
    mu_0: NumpyArray1D, Sigma_0: NumpyArray2D
) -> PriorInfo:
    """
    Arguments:
        mu_0: prior mean
        Sigma_0: prior variance
    """
    precision = np.linalg.inv(Sigma_0)
    precision_weighted_mean = precision @ mu_0
    return PriorInfo(precision, precision_weighted_mean)


###
# Sampling functions
##


def sample_omega(
    covariates: NumpyArray2D,
    beta_sample: NumpyArray2D,
    polya_gamma_sampler: pypolyagamma.pypolyagamma.PyPolyaGamma,
) -> NumpyArray1D:
    """
    Sample omega (the polya gamma augmentation variable) from the Polya Gamma distribution
    according to its complete conditional.

    Arguments:
        covariates: has shape (N,M),
            where N is the number of samples and M is the number of covariates
        beta_sample: has shape (M,K),
            where M is the number of covariates and K is the number of categories

    Returns:
        a sample of omega.  Has shape (N,K), where N is the number of samples and K is the number of categories
    """
    N, K = np.shape(covariates)[0], np.shape(beta_sample)[1]
    # TODO: the next two functions do a matrix computation twice, which is redundant
    eta = compute_linear_predictors_preventing_downstream_overflow(
        covariates, beta_sample
    )
    Z = compute_log_sum_exponentiated_utilities_for_non_self_categories(
        covariates, beta_sample
    )
    C = eta - Z
    omega_samples = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            omega_samples[n, k] = polya_gamma_sampler.pgdraw(1, C[n, k])
    return omega_samples


def sample_beta(
    labels: NumpyArray2D,
    covariates: NumpyArray2D,
    omega_sample: NumpyArray2D,
    beta_sample_previous: NumpyArray2D,
    prior_info: PriorInfo,
) -> NumpyArray1D:
    """
    Sample beta (the regression weights) according to its multivariate normal complete conditional.

    Arguments:
        labels: array with shape (N, K)
            where N is the number of samples and K is the number of categories
            A one-hot encoded representations of response categories.
        covariates: has shape (N,M), where N is the number of samples and M is the number of covariates
        omega_sample: has shape (N,K), where N is the number of samples and K is the number of categories
        beta_sample_previous: Has shape (M,K), where M is the number of covariates and K is the number of categories
            Used to

    Returns:
        a sample of beta. Has shape (M,K), where M is the number of covariates and K is the number of categories

    """
    M, K = np.shape(covariates)[1], np.shape(labels)[1]
    Z = compute_log_sum_exponentiated_utilities_for_non_self_categories(
        covariates, beta_sample_previous
    )

    beta_sample = np.zeros((M, K))
    for k in range(K):
        z_k = (labels[:, k] - 0.5) / omega_sample[:, k] + Z[:, k]
        W_k = scipy.sparse.diags(omega_sample[:, k])
        Sigma_k = np.linalg.inv(prior_info.precision + covariates.T @ W_k @ covariates)
        mu_k = Sigma_k @ (prior_info.precision_weighted_mean + covariates.T @ W_k @ z_k)
        beta_sample[:, k] = mvn.rvs(mu_k, Sigma_k)
    return beta_sample


###
# Main function
##


def sample_from_posterior_of_multiclass_logistic_regression_with_pga(
    covariates: NumpyArray2D,
    labels: NumpyArray1D,
    num_MCMC_samples: int,
    prior_info: Optional[PriorInfo] = None,
    beta_init: Optional[NumpyArray2D] = None,
) -> NumpyArray3D:
    """
    Sample from the posterior of the Multiclass Logistic Regression model with polya gamma augmentation.
    (Note that there is NO IB approximation here)

    Returns:
        np.array with shape (S,M,K) where S = num_MCMC_samples and the s-th row gives a sample of the beta matrix.
    """

    N, M = np.shape(covariates)
    K = np.shape(labels)[1]

    # initialize prior info
    if prior_info is None:
        M = np.shape(covariates)[1]
        mu_0 = np.zeros(M)
        Sigma_0 = np.eye(M)
        prior_info = prior_info_from_prior_params(mu_0, Sigma_0)

    # initialize sampler
    if beta_init is None:
        beta_init = np.zeros((M, K))
    beta_sample = beta_init

    # we construct this outside of the `sample_omega` function because each time PyPolyaGamma() is instantiated,
    # it resets the random seed.  And so if we were to instantiate this class each time we called `sample_omega`, then
    # we would get very similar samples whenever the parameters were similar, which is not the intended behavior
    # during Gibbs sampling.
    polya_gamma_sampler = PyPolyaGamma()

    # convert to dense if sparse
    # convert to dense if sparse
    if isspmatrix(covariates):
        covariates = np.array(covariates.todense())
    if isspmatrix(labels):
        labels = np.array(labels.todense(), dtype=int)

    beta_MCMC_samples = np.zeros((num_MCMC_samples, M, K))
    for s in range(num_MCMC_samples):
        END_OF_PRINT_STATEMENT = "\n"
        # "\r" is better if working locally, but won't show up in logs in cluster
        print(f"Now running MCMC iterate {s}", end=END_OF_PRINT_STATEMENT)
        sys.stdout.flush()

        sys.stdout.flush()
        omega_sample = sample_omega(covariates, beta_sample, polya_gamma_sampler)
        beta_sample_previous = beta_sample
        beta_sample = sample_beta(
            labels, covariates, omega_sample, beta_sample_previous, prior_info
        )

        # store what i want to retain
        beta_MCMC_samples[s, :, :] = beta_sample

    return beta_MCMC_samples
