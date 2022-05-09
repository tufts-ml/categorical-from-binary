from typing import Union

import numpy as np

from categorical_from_binary.types import NumpyArray1D, NumpyArray2D, NumpyArray3D


def beta_stds_from_beta_cov(
    beta_cov: Union[NumpyArray2D, NumpyArray3D]
) -> Union[NumpyArray1D, NumpyArray2D]:
    """
    Gets standard deviations of each covariate
        * If beta_cov is MxM, returns an array with shape (M,)
        * If beta_cov is MxMxK, returns an array with shape (M,K)
    """
    if beta_cov.ndim == 2:
        return np.sqrt(np.diag(beta_cov))
    elif beta_cov.ndim == 3:
        n_categories = np.shape(beta_cov)[2]
        return np.array(
            [np.sqrt(np.diag(beta_cov[:, :, k])) for k in range(n_categories)]
        ).T
    else:
        raise ValueError(
            "I don't know how to handle a beta covariance matrix of this dimensionality"
        )
