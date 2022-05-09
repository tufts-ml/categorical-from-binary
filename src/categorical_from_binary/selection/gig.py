"""
Generalized inverse gaussian helpers.
Uses same notation as in wikipedia 
"""


import numpy as np


np.set_printoptions(suppress=True, precision=3)

from scipy.special import kv as bessel2


def compute_expected_reciprocal_of_gig_random_variable_with_wikipedia_parameter_labeling(
    a: float,
    b: float,
    p: float,
) -> float:
    """
    Computes E[1/X], where X~GeneralizedInverseGaussian(a,b,p) with parameterization
    given in Wikipedia.
    """
    return (np.sqrt(a) / np.sqrt(b)) * (
        bessel2(p + 1, np.sqrt(a * b)) / bessel2(p, np.sqrt(a * b))
    ) - (2 * p / b)


def compute_expected_reciprocal_of_gig_random_variable_with_my_parameter_labeling(
    a: float,
    c: float,
    d: float,
) -> float:
    """
    Computes E[1/X], where X~GeneralizedInverseGaussian(a,b,p) with parameters labeled as in my
    categorical_from_binary report
    """
    return (np.sqrt(c) / np.sqrt(d)) * (
        bessel2(a + 1, np.sqrt(c * d)) / bessel2(a, np.sqrt(c * d))
    ) - (2 * a / d)
