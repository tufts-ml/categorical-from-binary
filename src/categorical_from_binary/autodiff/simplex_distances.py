import numpy as np

from categorical_from_binary.types import NumpyArray2D


def compute_mean_l1_distance(
    discrete_probs_1: NumpyArray2D, discrete_probs_2: NumpyArray2D
) -> float:
    """
    Compute the mean l1 norm.  Note the l1 norm is double the total variation distance.

    Arguments:
        discrete_probs_1: array with shape (N,K) where N is the number of samples and K is the
            number of categories
        discrete_probs_2: array with shape (N,K) where N is the number of samples and K is the
            number of categories
    """
    N, K = np.shape(discrete_probs_1)
    N2, K2 = np.shape(discrete_probs_2)

    if not N == N2 and K == K2:
        raise ValueError("Dimensionalities of discrete probability arrays must match")

    return np.sum(np.abs(discrete_probs_1 - discrete_probs_2)) / N
