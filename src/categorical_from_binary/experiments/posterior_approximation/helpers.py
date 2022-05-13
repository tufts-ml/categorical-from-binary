from dataclasses import dataclass
from typing import Dict

import numpy as np

from categorical_from_binary.data_generation.bayes_multiclass_reg import (
    Link,
    construct_category_probs,
)
from categorical_from_binary.types import NumpyArray1D, NumpyArray2D, NumpyArray3D


###
# Preliminaries
###


@dataclass
class BetaSamplesAndLink:
    samples: NumpyArray3D  # S x M x L
    link: Link


@dataclass
class CatProbData:
    feature_vector: NumpyArray1D
    samples: NumpyArray2D  # S x K
    link: Link


###
# Construct posterior samples (from optimization methods, IB-CAVI and ADVI, which don't yield samples)
###


def sample_beta_cavi(beta_mean, beta_cov_across_M_for_all_K, seed):
    """
    beta_mean: (M,K)
    beta_cov_across_M_for_all_K: (M,M)  (uniform across K)
    """
    np.random.seed(seed)

    beta_stds_across_M_for_all_K = np.sqrt(np.diag(beta_cov_across_M_for_all_K))
    M, K = np.shape(beta_mean)
    beta_matrix_sample = np.zeros_like(beta_mean)
    for k in range(K):
        beta_matrix_sample[:, k] = np.random.normal(
            loc=beta_mean[:, k], scale=beta_stds_across_M_for_all_K
        )
    return beta_matrix_sample


def sample_beta_advi(beta_mean, beta_stds, seed):
    """
    beta_mean: (M,L), where L in {K-1, K} depending on link function
    beta_stds: (M,L), where L in {K-1, K} depending on link function
    """
    np.random.seed(seed)

    M, L = np.shape(beta_mean)
    beta_matrix_sample = np.zeros_like(beta_mean)
    for m in range(M):
        for l in range(L):
            beta_matrix_sample[m, l] = np.random.normal(
                loc=beta_mean[m, l], scale=beta_stds[m, l]
            )
    return beta_matrix_sample


###
# Get category probability samples and variance for a FIXED covariate vector
###


def cat_prob_samples_from_beta_samples(
    feature_vector, beta_samples: NumpyArray3D, link: Link
) -> NumpyArray2D:
    """
    Arguments:
        beta_samples : array of shape (S,M,L)
            where S is the num of samples from the posterior, M is the num of covariates (incl intercept), and L
            is the number of "free" categories (i.e. those without an identifiability constraint imposed)
    """
    num_mcmc_samples, _, L = np.shape(beta_samples)
    K = np.shape(construct_category_probs(feature_vector, beta_samples[0, :, :], link))[
        1
    ]
    cat_prob_samples = np.zeros((num_mcmc_samples, K))
    for i in range(num_mcmc_samples):
        cat_prob_samples[i, :] = construct_category_probs(
            feature_vector, beta_samples[i, :, :], link
        )
    return cat_prob_samples


def construct_cat_prob_data_by_method(
    feature_vector, beta_samples_and_link_by_method: Dict[str, BetaSamplesAndLink]
) -> Dict[str, CatProbData]:
    cat_prob_data_by_method = dict()
    for method, beta_samples_and_link in beta_samples_and_link_by_method.items():
        bsl = beta_samples_and_link
        cat_prob_samples = cat_prob_samples_from_beta_samples(
            feature_vector, bsl.samples, bsl.link
        )
        cat_prob_data_by_method[method] = CatProbData(
            feature_vector, cat_prob_samples, bsl.link
        )
    return cat_prob_data_by_method
