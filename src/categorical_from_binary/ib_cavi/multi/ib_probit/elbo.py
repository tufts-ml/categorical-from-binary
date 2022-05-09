from typing import Union

import numpy as np
import scipy
from scipy.special import log_ndtr as log_norm_cdf_fast
from scipy.stats import norm

from categorical_from_binary.ib_cavi.binary_probit.elbo import (
    compute_variational_energy as compute_variational_energy_for_one_category,
    compute_variational_entropy_of_beta as compute_variational_entropy_of_beta_for_one_category,
)
from categorical_from_binary.math import logdiffexp
from categorical_from_binary.types import NumpyArray2D, NumpyArray3D


def _beta_cov_for_category_from_beta_cov(
    beta_cov: Union[NumpyArray2D, NumpyArray3D],
    category_index: int,
) -> NumpyArray2D:

    if np.ndim(beta_cov) == 2:
        return beta_cov
    elif np.ndim(beta_cov) == 3:
        return beta_cov[:, :, category_index]
    else:
        raise ValueError(
            "I am not sure how to get the beta covariance for each category"
        )


def compute_variational_entropy_of_z(
    labels: NumpyArray2D,
    design_matrix: NumpyArray2D,
    beta_mean: NumpyArray2D,
) -> NumpyArray2D:
    """
    Arguments:
        labels: array with shape (n_obs, n_categories)
            one-hot encoded representations of response categories
        design_matrix: array with shape (n_obs, n_features)
        beta_mean : array with shape (n_features, n_categories)
    """
    nu = design_matrix @ beta_mean
    # A more direct, but less numerically stable, way to code up what's happening here:
    # pdfs = norm.pdf(-nu)
    # cdfs = norm_cdf_fast(-nu)
    # mean_shift_if_all_categories_were_observed = pdfs / (1 - cdfs)
    # mean_shift_if_no_categories_were_observed = -pdfs / cdfs

    log_pdfs = norm.logpdf(-nu)
    log_cdfs = log_norm_cdf_fast(-nu)
    log_sfs = logdiffexp(0, log_cdfs)  # sf = 1 -cdf
    mean_shift_if_all_categories_were_observed = np.exp(log_pdfs - log_sfs)
    mean_shift_if_no_categories_were_observed = -np.exp(log_pdfs - log_cdfs)
    # TODO: the above was already calculated when computing expected value of z. Maybe borrow that calculation?
    C = np.log(np.sqrt(2 * np.pi * np.e))
    entropy_if_all_categories_were_observed = (
        C + log_sfs - nu * mean_shift_if_all_categories_were_observed / 2
    )
    entropy_if_no_categories_were_observed = (
        C + log_cdfs - nu * mean_shift_if_no_categories_were_observed / 2
    )
    return np.sum(
        labels * entropy_if_all_categories_were_observed
        + (1 - labels) * entropy_if_no_categories_were_observed
    )


def compute_variational_entropy_of_beta(
    beta_mean: NumpyArray2D,
    beta_cov: Union[NumpyArray2D, NumpyArray3D],
) -> float:
    """
    Arguments:
        beta_mean : array with shape (n_features, n_categories)
        beta_cov : Two possibilities:
                1) array with shape (n_features, n_features)
                    This is, at least with the current N(0,I) prior on each beta_k, identical for each of the
                    K categories, so we only store it once rather than K copies of it.
                2) array with shape (n_features, n_features, n_categories)
    """
    n_categories = np.shape(beta_mean)[1]
    entropy_of_beta = 0
    for k in range(n_categories):
        beta_mean_for_category = beta_mean[:, k]
        beta_cov_for_category = _beta_cov_for_category_from_beta_cov(beta_cov, k)
        entropy_of_beta += compute_variational_entropy_of_beta_for_one_category(
            beta_mean_for_category, beta_cov_for_category
        )
    return entropy_of_beta


def compute_variational_entropy(
    beta_mean: NumpyArray2D,
    beta_cov: Union[NumpyArray2D, NumpyArray3D],
    design_matrix: NumpyArray2D,
    labels: NumpyArray2D,
) -> float:

    entropy_z = compute_variational_entropy_of_z(labels, design_matrix, beta_mean)
    entropy_beta = compute_variational_entropy_of_beta(beta_mean, beta_cov)
    return entropy_z + entropy_beta


def compute_variational_energy(
    beta_mean: NumpyArray2D,
    beta_cov: Union[NumpyArray2D, NumpyArray3D],
    z_mean: NumpyArray2D,
    covariates: NumpyArray2D,
) -> float:
    n_categories = np.shape(beta_mean)[1]

    energy = 0
    for k in range(n_categories):
        beta_cov_for_category_k = _beta_cov_for_category_from_beta_cov(beta_cov, k)
        energy += compute_variational_energy_for_one_category(
            beta_mean[:, k],
            beta_cov_for_category_k,
            z_mean[:, k],
            covariates,
        )
    return energy


def compute_elbo(
    beta_mean: NumpyArray2D,
    beta_cov: Union[NumpyArray2D, NumpyArray3D],
    z_expected: NumpyArray2D,
    design_matrix: NumpyArray2D,
    labels: NumpyArray2D,
) -> float:
    """
    The independence structure of the model (and variational approximation),  combined with the one-vs-rest
    labeling strategy to make the model identifiable, makes the ELBO for the CBC-Probit model equal to the sum
    of K ELBOs from K separate binary probit regressions.

    Arguments:
        beta_mean : array with shape (n_features, n_categories)
        beta_cov : Two possibilities:
                1) array with shape (n_features, n_features)
                    This is, at least with the current N(0,I) prior on each beta_k, identical for each of the
                    K categories, so we only store it once rather than K copies of it.
                2) array with shape (n_features, n_features, n_categories)
        z_expected : array with shape (n_obs, n_categories)
        design_matrix: array with shape (n_obs, n_features)
        labels: array with shape (n_obs, n_categories)
            one-hot encoded representations of response categories
    """
    if scipy.sparse.issparse(design_matrix):
        # the multi-probit CBC inference code sometimes uses sparse matrix representations to speed up
        # computations of the form X'beta when the data sets are large.  But this representation breaks
        # the elbo code (specifically, at least `compute_variational_expectation_of_complete_data_likelihood`
        # in binary_probit.elbo, so for now we simply force back to dense.  An implication of this is that
        # ELBO computations will be slow for now on large datasets.)
        design_matrix = design_matrix.toarray()
    entropy = compute_variational_entropy(beta_mean, beta_cov, design_matrix, labels)
    energy = compute_variational_energy(
        beta_mean,
        beta_cov,
        z_expected,
        design_matrix,
    )
    return energy + entropy
