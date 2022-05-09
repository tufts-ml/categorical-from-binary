from typing import Union

import numpy as np

from categorical_from_binary.ib_cavi.binary_probit.elbo import (
    compute_elbo as compute_elbo_for_one_category,
)
from categorical_from_binary.types import NumpyArray2D, NumpyArray3D


def compute_elbo_intuitive(
    beta_mean: NumpyArray2D,
    beta_cov: Union[NumpyArray2D, NumpyArray3D],
    z_expected: NumpyArray2D,
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
        z_expected : array with shape (n_obs, n_categories)
        design_matrix: array with shape (n_obs, n_features)
        labels: array with shape (n_obs, n_categories)
            one-hot encoded representations of response categories
    """
    n_categories = np.shape(labels)[1]
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
            beta_mean[:, k],
            beta_cov_for_category,
            z_expected[:, k],
            design_matrix,
            labels[:, k],
            verbose=False,
        )
    return elbo
