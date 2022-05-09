import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pypolyagamma import PyPolyaGamma
from scipy.stats import multivariate_normal as mvn, norm

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


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


def sample_omega(
    covariates: NumpyArray2D, beta_sample: NumpyArray2D, z_sample: NumpyArray1D
) -> NumpyArray1D:
    """
    Sample omega (the polya gamma augmentation variable) from the Polya Gamma distribution
    according to its complete conditional.

    Arguments:
        covariates: has shape (N,M), where N is the number of samples and M is the number of covariates
        beta_sample: has shape (M,), where M is the number of covariates
        z_sample: has shape (N,), where N is the number of samples

    Returns:
        a sample of omega.  Has shape (N,), where N is the number of samples

    """
    b = 2
    c_per_sample = covariates @ beta_sample - z_sample
    pg = PyPolyaGamma()
    return np.array([pg.pgdraw(b, c) for c in c_per_sample])


def sample_beta(
    covariates: NumpyArray2D,
    z_sample: NumpyArray1D,
    omega_sample: NumpyArray1D,
    prior_info: PriorInfo,
) -> NumpyArray1D:
    """
    Sample beta (the regression weights) according to its multivariate normal complete conditional.

    Arguments:
        covariates: has shape (N,M), where N is the number of samples and M is the number of covariates
        omega_sample: has shape (N,), where N is the number of samples
        z_sample: has shape (N,), where N is the number of samples

    Returns:
        a sample of beta. Has shape (M,), where M is the number of covariates

    """
    Sigma = np.linalg.inv(
        prior_info.precision + covariates.T @ np.diag(omega_sample) @ covariates
    )
    mu = Sigma @ (
        prior_info.precision_weighted_mean
        + covariates.T @ np.diag(omega_sample) @ z_sample
    )
    return mvn.rvs(mu, Sigma)


def sample_z(
    covariates: NumpyArray2D,
    labels: NumpyArray1D,
    beta_sample: NumpyArray1D,
    omega_sample: NumpyArray1D,
) -> NumpyArray1D:
    """
    Sample z, the Diagonal Orthant augmentation variable (which in 2D is just the standard Albert & Chib augmentation
    variable), from its (truncated normal) complete conditional

    Arguments:
        covariates: has shape (N,M), where N is the number of samples and M is the number of covariates
        labels: array with shape (N,), whose elements are either a 1 or 0.
        beta_sample: has shape (M,), where M is the number of covariates
        omega_sample: has shape (N,), where N is the number of samples

    Returns:
        a sample of z. Has shape (N,), where N is the number of samples

    """
    N = np.shape(covariates)[0]

    mu_zs = covariates @ beta_sample
    ssq_zs = 1 / np.sqrt(omega_sample)

    z_samples = [None] * N
    # A crappy truncated normal sampler via rejection sampling.  Can do better if desired.
    for i in range(N):
        acceptance = False
        while not acceptance:
            sample_proposed = norm.rvs(mu_zs[i], ssq_zs[i])
            if (labels[i] == 0 and sample_proposed < 0) or (
                labels[i] == 1 and sample_proposed > 0
            ):
                acceptance = True
                z_samples[i] = sample_proposed
    return z_samples


def sample_from_posterior(
    covariates: NumpyArray2D,
    labels: NumpyArray1D,
    prior_info: PriorInfo,
    num_MCMC_samples: int,
    z_init: Optional[NumpyArray1D] = None,
    beta_init: Optional[NumpyArray1D] = None,
):
    """
    Sample from the posterior of the Binary Logistic Regression model with both diagonal orthant
    and polya gamma augmentation.  Note that in this special case of binary data, diagonal orthant
    augmentation is the same as Albert and Chib augmentation.
    """
    # initialize sampler
    N, M = np.shape(covariates)
    if z_init is None:
        z_init = np.zeros(N)
    if beta_init is None:
        beta_init = np.zeros(M)
    z_sample = z_init
    beta_sample = beta_init

    beta_MCMC_samples = np.zeros((M, num_MCMC_samples))
    for s in range(num_MCMC_samples):
        END_OF_PRINT_STATEMENT = "\n"
        # "\r" is better if working locally, but won't show up in logs in cluster
        print(f"Now running MCMC iterate {s}", end=END_OF_PRINT_STATEMENT)
        sys.stdout.flush()
        omega_sample = sample_omega(covariates, beta_sample, z_sample)
        beta_sample = sample_beta(covariates, z_sample, omega_sample, prior_info)
        z_sample = sample_z(covariates, labels, beta_sample, omega_sample)

        # store what i want to retain
        beta_MCMC_samples[:, s] = beta_sample

    return beta_MCMC_samples
