"""
Helpers related to the the tau's (the regression component-specific variances, aka, in the language
of Scott Poulson, the local shrinkage coefficients), in the case 
where one puts a normal-gamma prior on the regression coefficients
of a (S)CBC-Probit model in an attempt to allow sparsity. 

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

from categorical_from_binary.types import NumpyArray2D


np.set_printoptions(suppress=True, precision=3)


from categorical_from_binary.ib_cavi.binary_probit.inference.structs import VariationalTaus
from categorical_from_binary.selection.gig import (
    compute_expected_reciprocal_of_gig_random_variable_with_my_parameter_labeling,
)


def compute_expected_tau_reciprocal_array(
    variational_taus: VariationalTaus,
) -> NumpyArray2D:
    """
    Returns expected_tau_reciprocal_matrix. This bascially does `expected_reciprocal_of_tau_component`
    for all components simultaneously, and stores the results in a (diagonal) matrix.

    Returns:
        array with shape (M,K), where M is number of covariates and K is number of categories
    """
    qtaus = variational_taus

    M, K = np.shape(qtaus.d)
    expected_tau_reciprocal_array = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            expected_tau_reciprocal_array[
                m, k
            ] = compute_expected_reciprocal_of_gig_random_variable_with_my_parameter_labeling(
                qtaus.a[m, k], qtaus.c, qtaus.d[m, k]
            )
    return expected_tau_reciprocal_array
