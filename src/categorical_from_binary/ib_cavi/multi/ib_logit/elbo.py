from typing import Union

import numpy as np

from categorical_from_binary.polya_gamma.binary_logreg_vi.inference import (
    compute_elbo as compute_elbo_for_one_category,
)
from categorical_from_binary.types import NumpyArray2D, NumpyArray3D


def compute_elbo(
    beta_mean: NumpyArray2D,
    beta_cov: Union[NumpyArray2D, NumpyArray3D],
    design_matrix: NumpyArray2D,
    labels: NumpyArray2D,
) -> float:
    """
    This gives the intuitive procedure for computing the ELBO for do-probit. However, it is much slower than compute_elbo,
    so we do not use it in our main code.

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
        design_matrix: array with shape (n_obs, n_features)
        labels: array with shape (n_obs, n_categories)
            one-hot encoded representations of response categories
    """
    # TODO: Make faster version, analogous to the change from compute_elbo_intuitive to
    # compute_elbo in the ib_probit stuff

    n_categories = np.shape(labels)[1]
    n_covariates = np.shape(design_matrix)[1]
    prior_beta_mean = np.zeros(n_covariates)
    prior_beta_cov = np.eye(n_covariates)
    elbo = 0
    for k in range(n_categories):

        if np.ndim(beta_cov) == 2:
            beta_cov_for_category = beta_cov
        elif np.ndim(beta_cov) == 3:
            beta_cov_for_category = beta_cov[:, :, k]
        else:
            raise ValueError(
                "I am not sure how to get the beta covariance for each category"
            )

        elbo += compute_elbo_for_one_category(
            design_matrix,
            labels[:, k],
            beta_mean[:, k],
            beta_cov_for_category,
            prior_beta_mean,
            prior_beta_cov,
        )
    return elbo
