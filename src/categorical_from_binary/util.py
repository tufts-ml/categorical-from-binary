import warnings
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from categorical_from_binary.types import NumpyArray1D


def one_hot_encoded_array_from_categorical_indices(
    categorical_indices: NumpyArray1D,
    number_of_possible_categories: Optional[int] = None,
    sparse_representation: bool = False,
) -> np.array:
    """
    Takes a one-dim numpy array of integers and expands it to a two-dim numpy array
    that is one-hot encoded
    """

    # TODO: This function currently assumes that we should assign a column for every
    # integer within 0 and the maximal categorical index.   This should be related

    if number_of_possible_categories is None:
        # If `number_of_possible_categories` is not provided, we will infer it to be the maximum value
        # plus one, due to zero-indexing, but this potentially dangerous to do under the hood.  Is 0 a
        # legitimate value? If the max value is far higher than the number of possible values, do we really
        # want to represent everything between zero and the max?
        number_of_possible_categories = max(categorical_indices) + 1
        warnings.warn(
            f"Inferring the number of possible values to be {number_of_possible_categories}. "
            f"Does that seem corect?"
        )

    N = len(categorical_indices)
    K = number_of_possible_categories
    if sparse_representation:
        one_hot_matrix = csr_matrix((N, K))
    else:
        one_hot_matrix = np.zeros((N, K), dtype=int)

    one_hot_matrix[(np.arange(N), categorical_indices)] = 1
    return one_hot_matrix


def construct_standardized_design_matrix(design_matrix: np.array):
    return (design_matrix - np.mean(design_matrix, axis=0)) / np.std(
        design_matrix, axis=0
    )
