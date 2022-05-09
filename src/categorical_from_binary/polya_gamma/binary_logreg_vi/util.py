import numpy as np

from categorical_from_binary.types import NumpyArray2D


def compute_matrix_inverse(matrix: NumpyArray2D):
    # This is here as a standin; I might want to use caching to avoid
    # taking inverses of the same matrix (e.g. the prior covariance)
    # a million times.   Precomputing would be another option that would
    # be more straightforward code wise, but this changes the function signature
    # in ugly ways (e.g. we'd pass said function the prior PRECISION but variational cov... yuck!)
    return np.linalg.inv(matrix)


def compute_log_abs_det(matrix: NumpyArray2D):
    """
    Returns the log of the absolute value of the determinant of a matrix
    """
    sign_of_determinant, log_abs_det = np.linalg.slogdet(matrix)
    return log_abs_det
