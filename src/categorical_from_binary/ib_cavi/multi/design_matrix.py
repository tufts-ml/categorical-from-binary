import warnings
from typing import Union

import numpy as np
from scipy.sparse import spmatrix

from categorical_from_binary.types import NumpyArray2D


def construct_design_matrix(
    covariates: Union[NumpyArray2D, spmatrix],
    labels: Union[NumpyArray2D, spmatrix],
    use_autoregressive_design_matrix: bool = False,
) -> Union[NumpyArray2D, spmatrix]:
    """
    Construct a design matrix for a CBC model.

    If we are using an autoregressive model, we:
        * Ensure there is not already a column of all ones in the provided covariates matrix
        * Prepend one-hot encoded representation of the previous categories as predictors for the upcoming category.

    Note that the category labels are only actually needed to construct the design matrix if we are using an autoregressive
    model.

    Returns:
        A sparse matrix if `covariates` was sparse, else a dense np.array.
    """
    n_samples, n_categories = np.shape(labels)

    if not use_autoregressive_design_matrix:
        design_matrix = covariates
    else:
        # Covariates shouldnt have a vector of all ones for the autoregressive model.
        # If the model is autoregressive, we remove that column.
        first_column_of_covariates_is_all_ones = (
            covariates[:, 0]
            == np.ones(
                n_samples,
            )
        ).all()
        if first_column_of_covariates_is_all_ones:
            warnings.warn(
                f" The first column of covariates is all ones. We do not construct "
                f"the design matrix in this way for the autoregressive setting, because we decompose "
                f"the intercept term into K intercept terms, one for each of the K possible classes "
                f"that were inhabited at the preceding time step"
            )
            covariates_to_use = covariates[:, 1:]
        else:
            covariates_to_use = covariates

        # TODO: Automatically check that the code doesn't have
        n_covariates_to_use = np.shape(covariates_to_use)[1]

        # Construct previous labels, and prepend to covariates to make design matrix
        warnings.warn("Taking the initial category to be the smallest one.")
        label_init = np.eye(n_categories)[0]  # init label is arbitrary
        # TODO: need better handling for 0th label.... don't know what the previous label is ....
        # current init label scheme is going to be crappy... but using it for now.
        labels_prev = np.vstack((label_init, labels[:-1, :]))
        design_matrix = np.hstack((labels_prev, covariates_to_use))
        assert np.shape(design_matrix)[1] == n_categories + n_covariates_to_use
    return design_matrix
