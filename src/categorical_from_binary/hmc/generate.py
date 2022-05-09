"""
Generate data for the probit HMC demos and experiments

# TODO: Align with the categorical_from_binary.data_generation module
"""

"""
Purpose:
    Some helper functions for running experiments on CBC-Probit type models with HMC in the special case where
    everything is intercepts only.

Details:
    We want to infer betas, using HMC, for a variety of models (IB, CBC, CBM) and category probability formulas
    (CBC, SO).  We can study different samples sizes N , number of categories K,
    and true category probabilities p .. 

Reference: 
    https://github.com/tufts-ml/CBC-models/blob/main/experiments/CBCProbitDemo_K2.ipynb


Naming Conventions:
    I often use MCH's naming convention, which is to give the dimensionality of each array
    after the last underscore

# TODO: Support other links besides multi-logit (aka softmax)
"""

from typing import Tuple

import numpy as np
import scipy.special

from categorical_from_binary.util import one_hot_encoded_array_from_categorical_indices


def generate_categorical_data_with_covariates_using_multilogit_link(
    num_samples: int,
    num_categories: int,
    num_covariates_not_counting_bias: int,
    random_seed: int = 0,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Generate categorical data with covariates, using a multilogit (a.k.a softmax) link
    """
    # Using MCH's naming convention here, which is to give the dimensionality of each array
    # after the last underscore

    N = num_samples
    K = num_categories
    M = num_covariates_not_counting_bias + 1

    # Note: a numpy RandomState instance has methods for sampling from many distributions!
    # Reference: https://numpy.org/doc/1.16/reference/generated/numpy.random.RandomState.html
    # This seems like a stripped-down version of the distributions in scipy.stats (which seem
    # to provide more distributions, but also more methods, like mean, entropy, etc.).  However,
    # the parametrization in numpy seems more intuitive.
    prng = np.random.RandomState(random_seed)

    true_beta_KM = prng.randn(K, M)
    x_train_NM = np.hstack([prng.randn(N, M - 1), np.ones((N, 1))])
    x_test_NM = np.hstack([prng.randn(N, M - 1), np.ones((N, 1))])

    def calc_category_probs_NK(x_NM, true_beta_KM):
        """
        Note: This data generating process using a multi-logit (i.e. softmax) link
        """
        logp_NK = np.dot(x_NM, true_beta_KM.T)
        logp_NK -= scipy.special.logsumexp(logp_NK, axis=1, keepdims=1)
        return np.exp(logp_NK)

    p_train_NK = calc_category_probs_NK(x_train_NM, true_beta_KM)
    y_train_N = [prng.choice(np.arange(K), p=p_train_NK[n]) for n in range(N)]
    p_test_NK = calc_category_probs_NK(x_test_NM, true_beta_KM)
    y_test_N = [prng.choice(np.arange(K), p=p_test_NK[n]) for n in range(N)]

    y_train__one_hot_NK = one_hot_encoded_array_from_categorical_indices(y_train_N, K)
    y_test__one_hot_NK = one_hot_encoded_array_from_categorical_indices(y_test_N, K)
    return y_train__one_hot_NK, y_test__one_hot_NK, x_train_NM, x_test_NM


def generate_intercepts_only_categorical_data(
    true_category_probs_K: np.array,
    num_samples: int,
    random_seed: int = 0,
) -> Tuple[np.array, np.array]:
    """
    Generate (intercepts-only) categorical data
    """
    # Using MCH's naming convention here, which is to give the dimensionality of each array
    # after the last underscore

    K = len(true_category_probs_K)
    N = num_samples

    # Note: a numpy RandomState instance has methods for sampling from many distributions!
    # Reference: https://numpy.org/doc/1.16/reference/generated/numpy.random.RandomState.html
    # This seems like a stripped-down version of the distributions in scipy.stats (which seem
    # to provide more distributions, but also more methods, like mean, entropy, etc.).  However,
    # the parametrization in numpy seems more intuitive.
    prng = np.random.RandomState(random_seed)
    y_train_N = prng.choice(np.arange(K), p=true_category_probs_K, size=N, replace=True)
    y_test_N = prng.choice(np.arange(K), p=true_category_probs_K, size=N, replace=True)

    y_train__one_hot_NK = one_hot_encoded_array_from_categorical_indices(y_train_N, K)
    y_test__one_hot_NK = one_hot_encoded_array_from_categorical_indices(y_test_N, K)
    return y_train__one_hot_NK, y_test__one_hot_NK
