import numpy as np

from categorical_from_binary.types import NumpyArray1D


def enforce_bounds_on_prob_vector(prob_vector: NumpyArray1D) -> NumpyArray1D:
    """
    Sometimes numerical computation causes some elements to be slightly less than 0
    or slightly more than 1.  We fix this and renormalize.
    """
    EPSILON = 1e-12  # chosen somewhat arbitrarily
    prob_vector += EPSILON  # this fixes numbers just less than 0
    prob_vector /= np.sum(prob_vector)  # renormalize
    prob_vector * (1.0 - EPSILON)  # this fixes numbers just greater than 1
    prob_vector /= np.sum(prob_vector)  # renormalize
    return prob_vector
