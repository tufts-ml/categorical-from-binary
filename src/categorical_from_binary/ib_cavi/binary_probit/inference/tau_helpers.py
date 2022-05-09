"""
Helpers related to the the tau_m's (the regression component-specific variances, aka, in the language
of Scott Poulson, the local shrinkage coefficients), in the case 
where one puts a normal-gamma prior on the regression coefficients
in an attempt to allow sparsity. 

The normal-gamma prior is a generalization of the Laplace prior used for Bayesian Lasso.

We use closed-form coordinate ascent variational inference (CAVI) for inference.  I worked
out the update equations myself given the complete conditionals provdied by Brown
and Griffin (2010), Bayesian Analysis.  See below:

    @article{brown2010inference,
    title={Inference with normal-gamma prior distributions in regression problems},
    author={Brown, Philip J and Griffin, Jim E},
    journal={Bayesian analysis},
    volume={5},
    number={1},
    pages={171--188},
    year={2010},
    publisher={International Society for Bayesian Analysis}
    }

For more information, and notation, see my notes, located at:
    minis/sparsity_inducing_priors.pdf
"""


import numpy as np


np.set_printoptions(suppress=True, precision=3)

from scipy.special import kv as bessel2

from categorical_from_binary.ib_cavi.binary_probit.inference.structs import VariationalTaus
from categorical_from_binary.selection.hyperparameters import Hyperparameters


def compute_expected_reciprocal_of_tau_component(
    variational_tau_d_component: float, hyperparameters: Hyperparameters
) -> float:
    """
    Each regression coefficient beta_m has its own variance tau_m.  This computes the variational
    expectation of E[1/tau_m], which is needed in the coordinate ascent variational inference.

    Arguments:
        variational_tau_d_component: One component of the `d` vector from the
            VariationalTau object.
    """
    hp = hyperparameters
    qd_m = variational_tau_d_component
    return (bessel2(hp.lambda_ + 0.5, np.sqrt(qd_m) / hp.gamma)) / (
        hp.gamma * np.sqrt(qd_m) * (bessel2(hp.lambda_ - 0.5, np.sqrt(qd_m) / hp.gamma))
    ) - (2 * hp.lambda_ - 1) / (qd_m)


def compute_expected_tau_reciprocal_matrix(
    variational_taus: VariationalTaus, hyperparameters: Hyperparameters
):
    """
    Returns expected_tau_reciprocal_matrix. This bascially does `expected_reciprocal_of_tau_component`
    for all components simultaneously, and stores the results in a (diagonal) matrix.

    Arguments:
        lambda_ : hyperparameter for the normal-gamma prior on the regression weights
        gamma : hyperparameter for the normal-gamma prior on the regression weights
    """
    M = len(variational_taus.d)
    expected_tau_reciprocal_matrix = np.eye(M)
    for m in range(M):
        expected_tau_reciprocal_matrix[
            m, m
        ] = compute_expected_reciprocal_of_tau_component(
            variational_taus.d[m], hyperparameters
        )
    return expected_tau_reciprocal_matrix
