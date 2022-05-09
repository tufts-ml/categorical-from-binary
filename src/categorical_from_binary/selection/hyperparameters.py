"""
Hyperparameters for the gamma prior at the top of a normal-gamma prior on the regression 
weights, in the case where a continuous shrinkage prior for variable selection is used. 
"""

import numpy as np


np.set_printoptions(suppress=True, precision=3)

from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """
    Attributes:
        lambda_ : hyperparameter for the normal-gamma prior on the regression weights
        gamma : hyperparameter for the normal-gamma prior on the regression weights
    """

    lambda_: float
    gamma: float


def gamma_parameter_from_lambda_and_desired_marginal_beta_variance(
    variance: float, lambda_: float
) -> float:
    """
    Determine the gamma hyperparameter for a normal gamma prior, given:
        1) the other hyperparameter, lambda_, which controls the prior expectation on sparsity
        (lower values means more sparse)
        2) the desired variance in the regression weights
    """
    return np.sqrt(variance / (2 * lambda_))


def hyperparameters_from_lambda_and_desired_marginal_beta_variance(
    variance: float, lambda_: float
) -> Hyperparameters:
    gamma = gamma_parameter_from_lambda_and_desired_marginal_beta_variance(
        variance,
        lambda_,
    )
    return Hyperparameters(lambda_, gamma)
