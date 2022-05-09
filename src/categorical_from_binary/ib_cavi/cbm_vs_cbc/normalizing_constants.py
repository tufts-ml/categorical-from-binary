"""
A utility module for computing the normalizing constants (which in the workshop paper, we call "C")
for the complete data likelihoods of the CBC vs CBM models 
"""

import numpy as np
import scipy


np.set_printoptions(precision=3, suppress=True)

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D


def compute_log_Cs_CBC_probit(
    covariates: NumpyArray2D, beta: NumpyArray2D
) -> NumpyArray1D:
    """
    Returns the normalizing constant for the complete data likelihood of the
    CBC probit model. There is one for each observation.

    These are called "C" in the UAI-TPM workshop paper.
    """
    H = scipy.stats.norm.cdf
    X = covariates
    K = np.shape(beta)[1]
    N = np.shape(X)[0]
    log_Cs = np.zeros(N)
    for i in range(N):
        C_i = 0
        x_i = X[i, :]
        etas_i = x_i @ beta
        for k in range(K):
            signs = -np.ones(K)
            signs[k] = 1
            signed_etas_i = etas_i * signs
            C_i += np.prod(H(signed_etas_i))
        log_Cs[i] = np.log(C_i)
    return log_Cs


def compute_log_Cs_CBM_probit(
    covariates: NumpyArray2D, beta: NumpyArray2D, labels: NumpyArray2D
) -> NumpyArray1D:
    """
    Returns the normalizing constant for the complete data likelihood of the
    CBM probit model. There is one for each observation.

    These are called "C" in the UAI-TPM workshop paper.
    """
    choices = np.argmax(labels, 1)
    H = scipy.stats.norm.cdf
    X = covariates
    K = np.shape(beta)[1]
    N = np.shape(X)[0]
    log_Cs = np.zeros(N)
    for i in range(N):
        C_i = 0
        x_i = X[i, :]
        etas_i = x_i @ beta
        choice_id = choices[i]
        negative_etas_i_nonselected = -np.delete(etas_i, choice_id)
        for k in range(K):
            cdf_of_eta_for_current_category = H(etas_i[k])
            C_i += cdf_of_eta_for_current_category * np.prod(
                H(negative_etas_i_nonselected)
            )
        log_Cs[i] = np.log(C_i)
    return log_Cs


def compute_sum_of_IB_probs(
    covariates: NumpyArray2D, beta: NumpyArray2D
) -> NumpyArray1D:
    """
    The sum of IB probs is the the normalizing constant for the
    CBM's MARGINAL data likelihood (i.e. category probability formula, after
    marginalizing out the CBC variable "z").

    There is one for each observation.
    """
    H = scipy.stats.norm.cdf
    etas = covariates @ beta
    IB_probs = H(etas)
    return np.sum(IB_probs, 1)


def compute_log_sum_of_psis(
    covariates: NumpyArray2D, beta: NumpyArray2D
) -> NumpyArray1D:
    """
    The sum of psis is the the normalizing constant for the
    CBC's MARGINAL data likelihood (i.e. category probability formula, after
    marginalizing out the CBC variable "z"), when using the ALTERNATE expression
    (called "psi" in the UAI-TPM workshop paper and various reports by Mike H and I).

    We return the log because these can get VERY large.

    There is one for each observation.
    """
    H = scipy.stats.norm.cdf
    etas = covariates @ beta
    psis = H(etas) / H(-etas)  # psi is the name from the UAI-TPM workshop paper
    sum_of_psis = np.sum(psis, 1)
    return np.log(sum_of_psis)
