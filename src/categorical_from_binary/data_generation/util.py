from typing import Union

import numpy as np
import scipy
from scipy.sparse import spmatrix

from categorical_from_binary.types import NumpyArray2D


def prepend_features_with_column_of_all_ones_for_intercept(
    features: Union[NumpyArray2D, spmatrix]
) -> Union[NumpyArray2D, spmatrix]:
    """
    Returns:
        dense matrix if `features` is dense, sparse matrix if `features` is sparse
    """
    if scipy.sparse.issparse(features):
        sparse_representation = True
    else:
        sparse_representation = False

    n_samples = np.shape(features)[0]
    ones_vector = np.ones((n_samples, 1))
    if sparse_representation:
        # we add the `tocsc` method to convert to a csc sparse matrix.
        # otherwise we get a coo sparse matrix, which can not be indexed
        # in a row-like way as can csc matrices or np arrays (e.g. M[:10] returns an error
        # if M is a coo matrix)
        return scipy.sparse.hstack((ones_vector, features)).tocsc()
    else:
        return np.hstack((ones_vector, features))


def prod_of_columns_except(array, index_to_exclude):
    return np.prod(np.delete(array, index_to_exclude, axis=1), axis=1)


def construct_random_signs_as_integers(size: int):
    return 2 * np.random.randint(0, 2, size=size) - 1
